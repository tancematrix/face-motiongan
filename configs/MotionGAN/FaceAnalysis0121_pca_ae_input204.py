import itertools as __itertools

# Models
models = dict(
    generator = dict(
        model = 'PCAParam_generator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        w_dim = 64,
        use_z = 'transform',
        z_dim = 64,
        normalize_z = True,
        input_dim = 27
    ),
    discriminator = dict(
        model = 'MotionGAN_discriminator',
        top = 64,
        padding_mode = 'reflect',
        kw = 5,
        norm = 'spectral',
        use_sigmoid = True,
        use_pca = False
    ),
)

# Traiing strategy
train = dict(
    batchsize = 24,
    num_workers = 16,
    total_iterations = 100000,
    out = 'results/MotionGAN/FaceAnalysis0118_ae_input204_2',

    # Dataset
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize/train_1',
        pca_root = '/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize',
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
    preview_interval = 5000,
    save_interval = 5000,

    # Loss
    GAN_type = 'normal',
    trjloss_sampling_points = 3,
    parameters=dict(
        g_lr = 0.0002,
        d_lr = 0.0001,
        lam_g_adv = 1.,
        lam_g_trj = 0,
        lam_g_cls = 1,
        lam_g_cons = 5000.,
        lam_d_adv = 1,
        lam_d_cls = 1,
    ),

    # Preview video parameters
    preview=dict(
        view_range = 50,
        save_delay = 4,
    ),
)

# Testing strategy
test = dict(
    out = 'results/MotionGAN/FaceAnalysis0118_ae_input204_2',
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize/test_1',
        pca_root = '/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize',
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
    out = 'results/MotionGAN/FaceAnalysis0118_ae_input204_2',
    dataset=dict(
        data_root = '/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize/test',
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
