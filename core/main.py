import core.process as process
import core.ocr_operation as ocr_operation


def c_main(path, num, ext):
    image_data = process.pre_process(path, num, ext)
    return image_data[1] + '.' + ext


def ocr_main(path):
    res = ocr_operation.detect(path)
    print("res=", res)
    # ocr_text = "ocr text"
    ocr_text = res
    return ocr_text


if __name__ == '__main__':
    pass


