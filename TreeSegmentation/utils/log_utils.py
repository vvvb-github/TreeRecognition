color_table = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'purple': 35,
    'cyan': 36,
    'white': 37
}


def log_success(format):
    print('\033[0;{}m{}\033[0m'.format(32, format))


def log_warn(format):
    print('\033[0;{}m{}\033[0m'.format(33, format))


def log_error(format):
    print('\033[0;{}m{}\033[0m'.format(31, format))


def log_info(format):
    print('\033[0;{}m{}\033[0m'.format(37, format))


def log_highlight(format):
    print('\033[0;{}m{}\033[0m'.format(36, format))
