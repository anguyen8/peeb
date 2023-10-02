import json
import os
import re
import openai
from tqdm import tqdm
from utils import PROJECT_ROOT, load_descriptions
import tiktoken
import time
from ast import literal_eval

openai.api_key = os.getenv("OPENAI_API_KEY")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-3.5-turbo-0301", "gpt-4"]:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def generate_prompt_sachit(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""


def generate_descriptors(dataset, model):

    if dataset in ["cub", "cub_sachit"]:
        templated_descriptions, _ = load_descriptions(dataset_name="cub", prompt_type=0)
        class_names = list(templated_descriptions.keys())
    elif dataset in ["inaturalist", "inaturalist_sachit"]:
        with open("../data/class_names/inaturalist_classes.json", "r") as input_file:
            class_names = json.load(input_file)
    elif dataset in ["nabirds", "nabirds_sachit"]:
        with open("../data/class_names/nabirds_classes.json", "r") as input_file:
            class_names = json.load(input_file)
    elif dataset == "bird_soup":
        with open("../data/class_names/bird_soup_uncased_v2_classes.txt", "r") as input_file:
            class_names = [class_name.strip() for class_name in input_file.readlines()]
    elif dataset == "ebird":
        with open("../data/class_names/ebird_classes.txt", "r") as input_file:
            class_names = [class_name.strip() for class_name in input_file.readlines()]
    elif dataset == "bird_soup_scientific":
        with open("../data/class_names/cub_nabird_inat_sci.txt", "r") as input_file:
            class_names = [class_name.strip() for class_name in input_file.readlines()]

    if dataset in ["cub", "inaturalist", "nabirds", "bird_soup", "ebird", "bird_soup_extra", "bird_soup_scientific"]:
        content = "A bird has 12 parts: back, beak, belly, breast, crown, forehead, eyes, legs, wings, nape, tail and throat. " \
                  "Visually describe all parts of {} bird in a short phrase in bullet points using the format 'part: short phrase'"
    else:
        raise Exception(f"Dataset {dataset} is not supported")

    save_dir = f"{PROJECT_ROOT}/data/descriptors/chatgpt/raw/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    raw_descriptors_fp = f"{save_dir}/descriptors_{dataset}.json"
    descriptors_fp = f"{save_dir.replace('raw/', '')}/descriptors_{dataset}.json"
    chat_results = {}
    all_messages = []

    if os.path.exists(raw_descriptors_fp):
        with open(raw_descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

    loaded_class_names = list(chat_results.keys())

    # overwriten_class_names = ["tool kit", "mask", "necklace", "scarf", "website", "menu"]
    # overwriten_class_names = ["butterfly coquette", "cerulean cuckooshrike", "eye ringed flatbill", "magpie tanager", "red wattled lapwing"]
    overwriten_class_names = []

    index = 0
    with tqdm(total=len(class_names)) as pbar:
        while index < len(class_names):
            if class_names[index] in loaded_class_names and class_names[index] not in overwriten_class_names:
                index += 1
                pbar.update(1)
                continue

            try:
                if dataset in ["cub_sachit", "nabirds_sachit", "inaturalist_sachit"]:
                    content = generate_prompt_sachit(class_names[index])
                    temperature = 0.7
                else:
                    content = content.format(class_names[index], class_names[index])
                    temperature = 1.0

                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "user",
                         "content": content}
                    ],
                    temperature=temperature)
            except Exception as e:
                print(e)
                time.sleep(60)
                continue

            chat_results[class_names[index]] = completion.choices[0].message["content"]
            all_messages.append(completion.choices[0].message)

            num_tokens = num_tokens_from_messages([completion.choices[0].message], model)
            print(f"Number of tokens: {num_tokens}\n")

            with open(raw_descriptors_fp, "w") as output_file:
                json.dump(chat_results, output_file, indent=4)

            index += 1
            pbar.update(1)

    if not os.path.exists(descriptors_fp):
        with open(raw_descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

        for class_name, chat_result in chat_results.items():
            descriptors = []
            for descriptor in chat_result.strip().replace("- ", "").split("\n"):
                if dataset not in ["cub_sachit", "nabirds_sachit", "inaturalist_sachit"]:
                    descriptor = re.sub(r"^\W+", "", descriptor.strip().lower())
                    descriptor = re.sub(r"\W+$", "", descriptor)

                if (descriptor and ":" in descriptor) or dataset in ["cub_sachit", "nabirds_sachit", "inaturalist_sachit"]:
                    descriptors.append(descriptor)
            chat_results[class_name] = descriptors

        with open(descriptors_fp, "w") as output_file:
            json.dump(chat_results, output_file, indent=4)
    else:
        with open(descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

    total_num_tokens = 0
    for message in all_messages:
        total_num_tokens += num_tokens_from_messages([message], model)
    print(f"Total number of tokens: {total_num_tokens}\n")


def correct_cub_descriptors(model):

    with open(f"{PROJECT_ROOT}/data/descriptors/chatgpt/cub_allaboutbird_desc.json", "r") as input_file:
        cub_correct_descriptors = json.load(input_file)
        class_names = list(cub_correct_descriptors.keys())

    with open(f"{PROJECT_ROOT}/data/descriptors/chatgpt/descriptors_bird_soup_v21.json", "r") as input_file:
        cub_descriptors = json.load(input_file)

    # with open(f"{PROJECT_ROOT}/data/descriptors/chatgpt/descriptors_cub_sachit.json", "r") as input_file:
    #     cub_descriptors = json.load(input_file)
    #     cub_descriptors = {k.lower().replace("'s", "").replace("-", " ").replace("_", " ").replace("forsters tern", "forster tern"): v for k, v in cub_descriptors.items()}

    content = """Given the description of the bird "{}" as follows:

{}

Please summarize the visual features. Then check the items below one by one, if the item does not match or cannot be inferred from the description, correct it. If no information is provided, use your own knowledge.

{}

Please return with the same format. Note that if not specified, simply keep the original description.
    """

    save_dir = f"{PROJECT_ROOT}/data/descriptors/chatgpt/raw"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    raw_descriptors_fp = f"{save_dir}/descriptors_cub_auto_correct.json"                    # _sachit
    descriptors_fp = f"{save_dir.replace('/raw', '')}/descriptors_cub_auto_correct.json"    # _sachit
    chat_results = {}
    all_messages = []

    if os.path.exists(raw_descriptors_fp):
        with open(raw_descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

    loaded_class_names = list(chat_results.keys())

    overwriten_class_names = []
    index = 0
    questions = {}

    for idx, class_name in enumerate(class_names):
        old_descs = "[\n"
        for item in cub_descriptors[class_name]:
            old_descs += "\t\"" + item + "\",\n"
        old_descs += "]"

        description = content.format(class_name, cub_correct_descriptors[class_name], old_descs)
        questions[class_name] = description

    with open("questions.json", "w") as output_file:                                        # _sachit
        json.dump(questions, output_file, indent=4)

    # for idx, class_name in tqdm(enumerate(class_names)):
    with tqdm(total=len(class_names)) as pbar:
        while index < len(class_names):
            if class_names[index] in loaded_class_names and class_names[index] not in overwriten_class_names:
                index += 1
                pbar.update(1)
                continue

            try:
                old_descs = "[\n"
                for item in cub_descriptors[class_names[index]]:
                    old_descs += "\t\"" + item + "\",\n"
                old_descs += "]"

                description = content.format(class_names[index], cub_correct_descriptors[class_names[index]], old_descs)
                temperature = 1.0
                if index < 5:
                    print(description)

                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "user",
                         "content": description}
                    ],
                    temperature=temperature)
            except Exception as e:
                print(e)
                time.sleep(60)
                continue

            chat_results[class_names[index]] = completion.choices[0].message["content"]
            all_messages.append(completion.choices[0].message)

            num_tokens = num_tokens_from_messages([completion.choices[0].message], model)
            print(f"Number of tokens: {num_tokens}\n")

            with open(raw_descriptors_fp, "w") as output_file:
                json.dump(chat_results, output_file, indent=4)

            index += 1
            pbar.update(1)

    if not os.path.exists(descriptors_fp):
        with open(raw_descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

        for class_name, chat_result in chat_results.items():
            chat_results[class_name] = literal_eval(chat_result)

        with open(descriptors_fp, "w") as output_file:
            json.dump(chat_results, output_file, indent=4)
    else:
        with open(descriptors_fp, "r") as input_file:
            chat_results = json.load(input_file)

    total_num_tokens = 0
    for message in all_messages:
        total_num_tokens += num_tokens_from_messages([message], model)
    print(f"Total number of tokens: {total_num_tokens}\n")


def test_chatgpt(model, content):
    print(content)
    print(f"Response from ChatGPT version {model}:")

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user",
             "content": content}
        ])

    print(completion.choices[0].message["content"])

    num_tokens = num_tokens_from_messages([completion.choices[0].message], model)
    print(f"Number of tokens: {num_tokens}\n")

    return num_tokens


if __name__ == '__main__':
    model = "gpt-4"  # gpt-3.5-turbo, gpt-4

    generate_descriptors(dataset="cub", model=model)
    # generate_descriptors(dataset="nabirds", model=model)
    # generate_descriptors(dataset="bird_soup_extra", model=model)
    # generate_descriptors(dataset="bird_soup_scientific", model=model)
    # correct_cub_descriptors(model=model)
