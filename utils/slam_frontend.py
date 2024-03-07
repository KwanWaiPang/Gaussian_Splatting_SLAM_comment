import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True #系统是否要重置，初始化后设置为 False。
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1 #每隔多少帧使用一次

        self.gaussians = None
        self.cameras = dict() #保存所有的相机视角
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"] #从配置中获取 RGB 边界阈值。
        # 将当前帧索引添加到关键帧索引列表中。
        self.kf_indices.append(cur_frame_idx)
        # 获取当前帧的视角信息。
        viewpoint = self.cameras[cur_frame_idx]
        # 获取当前视角的原始图像 gt_img。用于进行训练的吧？
        gt_img = viewpoint.original_image.cuda()
        # 计算图像中 RGB 像素值的和，然后与 RGB 边界阈值进行比较，得到有效的 RGB 像素值。
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:#如果系统是单目系统
            if depth is None:#如果不提供深度图
                # 初始化深度为一个固定值，并添加一些噪声
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                # 根据深度和不透明度信息计算初始深度，并根据有效 RGB 像素值进行掩码。
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:#如果使用逆深度，上面给定为false
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else: #根据深度和不透明度信息计算初始深度，并根据有效 RGB 像素值进行掩码。
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0] #将深度数据转换为 NumPy 数组并返回。
        # use the observed depth（若不是单目，那么就使用观测到的深度信息）
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0) #从视角中获取深度信息，并将其转换为张量。
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels （略无效的 RGB 像素值，并将它们的深度设置为零。）
        return initial_depth[0].numpy() #将深度数据转换为 NumPy 数组并返回。

    # 初始化SLAM系统
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular #如果系统不是单目（monocular），则将初始化状态设置为 True，否则为 False。
        self.kf_indices = [] #初始化关键帧索引列表为空。
        self.iteration_count = 0 #初始化迭代计数器为 0。
        self.occ_aware_visibility = {} #初始化一个空字典，用于存储每个关键帧的可见性信息。
        self.current_window = [] #初始化当前窗口为空，该窗口用于跟踪一系列关键帧。

        # remove everything from the queues (清空队列中的数据：通过一个循环清空后端队列中的所有数据，以确保系统处于干净的状态。)
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        # 将当前视角的旋转和平移更新为gt真实姿态。(并放到gpu上)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = [] #再次将关键帧索引列表清空。
        # 添加新的关键帧，并生成深度地图。
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map) #向后端请求初始化，传递当前帧索引、视角信息和深度地图。
        self.reset = False #重置标志位为 False，表示系统不需要重置。

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames] #从当前帧往回倒退 self.use_every_n_frames(设置为1就是每帧都使用) 帧，获取前一帧的相机信息作为参考。
        viewpoint.update_RT(prev.R, prev.T) #使用前一帧的旋转矩阵和平移向量来更新当前帧的相机旋转和平移。（此时为上一帧的pose）

        opt_params = []#创建一个空列表，用于存储优化器的参数。
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        # 用 Adam 优化器来优化相机的姿态参数。
        pose_optimizer = torch.optim.Adam(opt_params)
        # 循环执行跟踪迭代次数。
        for tracking_itr in range(self.tracking_itr_num):
            #  调用 render 函数，生成渲染的图像、深度和不透明度信息。
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad() #梯度清零。
            # 计算跟踪过程中的损失函数。
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()#反向传播，计算梯度。

            with torch.no_grad():
                pose_optimizer.step() #更新参数，尝试使损失函数最小化。
                converged = update_pose(viewpoint) #更新相机的姿态。

            if tracking_itr % 10 == 0: #每隔10次迭代，将当前帧的信息传递给gui。发送到 q_main2vis 队列中，用于可视化。
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged: #如果收敛了，就退出
                break

        self.median_depth = get_median_depth(depth, opacity) #计算深度图的中值深度。
        return render_pkg #返回渲染包（render_pkg），其中包含了渲染的图像、深度和不透明度。

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    # 这个方法将传递的数据中的高斯模型、可见性信息和关键帧信息分别赋值给前端的对应属性。然后遍历关键帧信息列表，对于每个关键帧，更新相应的相机参数。
    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0 #初始化当前帧的索引为 0
        # 获取投影矩阵（三维点到像素坐标系上）
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device) #将投影矩阵转移到GPU上
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty(): #如果gui队列为空
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"]) #如果gui暂停了，那么就通知后端暂停
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty(): #如果前端队列为空
                tic.record() #记录当前时间，用于计算处理时间。
                if cur_frame_idx >= len(self.dataset): #如果当前帧的索引大于数据集的长度，也就是遍历完了~
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                #检查是否有初始化请求
                if self.requested_init: 
                    time.sleep(0.01)
                    continue
                
                # 检查是否处于单线程模式且有请求的关键帧。
                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                # 检查是否未初始化且有请求的关键帧。
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                
                #从数据集中获取当前帧的图像、深度图和位姿等数据(viewpoint)。 
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config) #计算梯度掩码

                # 将当前帧的视角（viewpoint）保存到 self.cameras 中，以便后续使用。
                self.cameras[cur_frame_idx] = viewpoint

                # 如果需要重置系统，执行以下操作：
                if self.reset:#初始化后设置为 False。
                    self.initialize(cur_frame_idx, viewpoint) #使用当前帧初始化系统
                    self.current_window.append(cur_frame_idx) #将当前帧索引添加到窗口中，窗口可能用于跟踪一系列关键帧。
                    cur_frame_idx += 1
                    continue
                
                # 如果 self.initialized 已经被设置为真（即已经初始化），那么它的值将保持不变；
                # 如果 self.initialized 尚未被设置为真，但当前窗口中的帧数等于指定的窗口大小，则将 self.initialized 的值设置为真。
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint) #似乎是获取渲染的结果

                current_window_dict = {} #创建一个空字典，用于存储当前窗口的关键帧。
                # 将当前窗口的关键帧存储到字典中，键为当前窗口的第一个帧，值为除第一个帧之外的其余帧。
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                # 据当前窗口的关键帧索引，获取对应的关键帧摄像机信息。
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # 将高斯包装对象放入队列 q_main2vis 中，用于可视化。这个包装对象包含克隆的高斯模型、当前帧、关键帧列表和当前窗口的字典。
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                
                # 如果有请求的关键帧。
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx) #清理当前帧
                    cur_frame_idx += 1 #当前帧索引加一。
                    continue #跳过当前循环，继续执行下一次循环。

                last_keyframe_idx = self.current_window[0] #获取当前窗口的第一个关键帧索引。
                # 计算当前帧与上一个关键帧之间的时间间隔是否大于等于关键帧间隔。
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                # 获取当前帧的可见性？？？
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # 根据一些条件判断是否创建关键帧，这些条件包括当前帧索引、上一个关键帧索引、当前帧的可见性以及其他一些参数。
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )

                # 如果当前窗口的帧数小于指定的窗口大小，则将当前帧添加到窗口中。
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算当前帧可见性和上一个关键帧的可见性的并集中非零元素的数量。
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算当前帧可见性和上一个关键帧的可见性的交集中非零元素的数量。
                    point_ratio = intersection / union #计算交集与并集的比值，表示当前帧在上一个关键帧的可见性范围内的点所占的比例。
                    # 判断是否需要创建关键帧，条件是当前帧与上一个关键帧之间的时间间隔大于等于关键帧间隔，并且点的比例小于指定的阈值 kf_overlap。
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )

                # 如果是单线程模式
                if self.single_thread:
                    create_kf = check_time and create_kf #如果是单线程模式，并且满足时间间隔条件，那么就需要创建关键帧。这段代码的作用是确保在单线程模式下，即使点的比例也符合要求，依然需要创建关键帧。

                # 如果需要创建关键帧 
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )#调用 add_to_window 方法，将当前帧添加到当前窗口中，并返回更新后的当前窗口和已移除的关键帧（如果有的话）。
                    # 如果是单目摄像头且地图尚未初始化且已移除了关键帧，则执行以下操作。
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True #将重置标志设置为True。因为如果地图尚未初始化且已移除了关键帧，那么就需要重置系统，也即需要重新初始化。
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    ) #调用 add_new_keyframe 方法，根据渲染包的深度和不透明度信息添加新的关键帧。
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    ) #请求添加关键帧。
                else: #如果不需要创建关键帧，那么就cleanup
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx) #进行ATE评估，并输出当前frame的索引。
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:#如果前端队列不为空
                data = self.frontend_queue.get() #从前端队列中获取数据。

                # 如果数据的第一个元素是 "sync_backend"，则执行以下操作：
                if data[0] == "sync_backend":
                    self.sync_backend(data) #调用 sync_backend 方法，将获取到的数据作为参数传递给该方法。

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
