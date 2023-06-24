from core import process


def c_main(path, num, ext):
    image_data = process.pre_process(path, num, ext)
    return image_data[1] + '.' + ext


def ocr_main(path, ext):
    return process.ocr_process(path, ext)


if __name__ == '__main__':
    pass


