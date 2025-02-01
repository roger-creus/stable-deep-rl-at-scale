def parse_network_size(net_size):
    if net_size not in ["small", "medium", "large", "default"]: raise ValueError(f"Unknown network size: {net_size}")
    if net_size == "small":
        return (16, 32, 32), 512, 1
    elif net_size == "medium":
        return (32, 64, 64), 1024, 2
    elif net_size == "large":
        return (64, 128, 128), 2048, 3
    elif net_size == "default":
        return (32, 64, 64), 512, 1
