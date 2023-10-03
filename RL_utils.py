import torch


def sample_action_and_logprob(action_mean, action_std):
    # action_std 값을 제한
    action_std = torch.clamp(action_std, min=0.1, max=2.0)

    # 정규 분포 객체 생성
    normal_distribution = torch.distributions.Normal(action_mean, action_std)

    # 정규 분포에서 샘플링
    action = normal_distribution.sample()

    # log probability 계산
    log_prob = normal_distribution.log_prob(action).sum(dim=-1)

    return action, log_prob


def add_brightness_to_batch_images(images, actions):
    # 이미지의 크기를 가져옴
    batch_size, _, height, width = images.shape

    # action을 sigmoid 함수를 통과시킴
    actions = torch.sigmoid(actions)

    # action에서 x, y 좌표와 brightness 값을 가져옴
    x = (actions[:, 0] * (28 - 2)).int()
    y = (actions[:, 1] * (28 - 2)).int()
    brightness = ((actions[:, 2] * 255).int()).float() / 255

    new_images = images.clone()
    for i in range(batch_size):
        new_images[i, 0, y[i] : y[i] + 1, x[i] : x[i] + 1] = brightness[i]

    actions = torch.stack([x, y, brightness], dim=-1)
    return actions, new_images
