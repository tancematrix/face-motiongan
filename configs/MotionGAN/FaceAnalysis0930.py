import itertools as __itertools

# Models
models = dict(
    generator = dict(
        model = 'MotionGAN_generator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        w_dim = 64,
        use_z = 'transform',
        z_dim = 64,
        normalize_z = True,
    ),
    discriminator = dict(
        model = 'MotionGAN_discriminator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        norm = 'spectral',
        use_sigmoid = True,
    ),
)

# Traiing strategy
train = dict(
    batchsize = 24,
    num_workers = 16,
    total_iterations = 50000,
    out = 'results/MotionGAN/FaceAnalysis1005',

    # Dataset
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/Homography2/train',
        multi_class = True,
        class_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        start_offset = 1,
        control_point_interval = 8,
        standard_bvh = None,
        scale = 1.,
        frame_nums = 64,
        frame_step = 2,
        augment_fps = True,
        rotate = False,
        head_rotation = True
    ),

    # Iteration intervals
    display_interval = 100,
    preview_interval = 1000,
    save_interval = 5000,

    # Loss
    GAN_type = 'normal',
    trjloss_sampling_points = 3,
    parameters=dict(
        g_lr = 0.0002,
        d_lr = 0.0001,
        lam_g_adv = 1.,
        lam_g_trj = 1.,
        lam_g_cls = 0.1,
        lam_d_adv = 1,
        lam_d_cls = 0.1,
    ),

    # Preview video parameters
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)

# Testing strategy
test = dict(
    out = 'results/MotionGAN/FaceAnalysis1005',
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/Rotation/test',
        class_list = [],
        start_offset = 1,
        control_point_interval = 8,
        standard_bvh = None,
        scale = 1.,
        frame_step = 2,
        head_rotation = True,
    ),
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)

evalc = dict(
    out = 'results/MotionGAN/FaceAnalysis1005',
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/Homography2/test',
        class_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        start_offset = 1,
        control_point_interval = 8,
        standard_bvh = None,
        scale = 1.,
        frame_nums = 64,
        frame_step = 2,
        head_rotation = True,
    ),
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)
