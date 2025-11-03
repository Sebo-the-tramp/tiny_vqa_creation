import os
import io
import os.path as osp
import base64
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1e9


def rescale_img(img, tgt=None):
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img


def encode_image_to_base64(img, target_size=-1, fmt="JPEG"):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode("utf-8")
    max_size = os.environ.get("VLMEVAL_MAX_IMAGE_SIZE", 1e9)
    min_edge = os.environ.get("VLMEVAL_MIN_IMAGE_EDGE", 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode("utf-8")

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode("utf-8")

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f"Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, "
            f"resize to {factor:.2f} of original size: ({new_w}, {new_h})"
        )

    return ret


def encode_image_file_to_base64(image_path, target_size=-1, fmt="JPEG"):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size, fmt=fmt)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)


def build_option_str(option_dict):
    s = "There are several options: \n"
    for c, content in option_dict.items():
        # if not pd.isna(content):
        s += f"{c}. {content}\n"
    return s


def isimg(s):
    return osp.exists(s) or s.startswith("http")


def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False


def gpt_key_set():
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_key is None:
        openai_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        return isinstance(openai_key, str)
    return isinstance(openai_key, str) and openai_key.startswith("sk-")


def apiok(wrapper):
    s = wrapper.generate("Hello!")
    return wrapper.fail_msg not in s
