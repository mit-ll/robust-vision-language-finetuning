'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Get architecture filename and associated loader according to dataset

from typing import Callable, Optional

from fmdg._data import get_loaders_main_224, get_loaders_CLIP

__all__ = [
    "text_prefix",
    "text_suffix",
    "loader_funcs",
    "get_data",
    "DGData",
    "convert_arch_to_filename",
]

text_prefix = {
    "camelyon17": "this is a photo of ",
    "fmow": "a centered satellite photo of ",
    "iwildcam": "this is a photo of ",
}

text_suffix = {
    "camelyon17": ["healthy lymph node tissue", "lymph node tumor tissue"],
    "fmow": [
        "airport",
        "airport hangar",
        "airport terminal",
        "amusement park",
        "aquaculture",
        "archaeological site",
        "barn",
        "border checkpoint",
        "burial site",
        "car dealership",
        "construction site",
        "crop field",
        "dam",
        "debris or rubble",
        "educational institution",
        "electric substation",
        "factory or powerplant",
        "fire station",
        "flooded road",
        "fountain",
        "gas station",
        "golf course",
        "ground transportation station",
        "helipad",
        "hospital",
        "impoverished settlement",
        "interchange",
        "lake or pond",
        "lighthouse",
        "military facility",
        "multi-unit residential",
        "nuclear powerplant",
        "office building",
        "oil or gas facility",
        "park",
        "parking lot or garage",
        "place of worship",
        "police station",
        "port",
        "prison",
        "race_track",
        "railway bridge",
        "recreational facility",
        "road bridge",
        "runway",
        "shipyard",
        "shopping mall",
        "single-unit residential",
        "smokestack",
        "solar farm",
        "space facility",
        "stadium",
        "storage tank",
        "surface mine",
        "swimming pool",
        "toll booth",
        "tower",
        "tunnel opening",
        "waste disposal",
        "water treatment facility",
        "wind farm",
        "zoo",
    ],
    "iwildcam": [
        "empty",
        "White-lipped peccary",
        "Central American agouti",
        "Lowland paca",
        "Cougar",
        "South American tapir",
        "Collared peccary",
        "Red brocket",
        "Ocelot",
        "Ruddy quail-dove",
        "South American coati",
        "Nine-banded armadillo",
        "Tayra",
        "Common opossum",
        "Crab-eating raccoon",
        "Jaguar",
        "Giant anteater",
        "Great tinamou",
        "Tapeti",
        "Jaguarundi",
        "Margay",
        "Brown Brocket",
        "Gray four-eyed opossum",
        "Wild goat",
        "Cattle Cow Bull",
        "Sheep",
        "Wolf",
        "Scrub hare",
        "Olive baboon",
        "Common genet",
        "Cape bushbuck",
        "Slender mongoose",
        "African bush elephant",
        "Gambian pouched rat",
        "Steenbok",
        "Striped hyena",
        "Impala",
        "Spotted hyena",
        "Caracal",
        "Wild horse",
        "Lion",
        "Common eland",
        "Waterbuck",
        "Common warthog",
        "Leopard",
        "White-tailed mongoose",
        "Black-backed jackal",
        "African buffalo",
        "Plains zebra",
        "Giraffe",
        "Hartebeest",
        "Vervet monkey",
        "Günther's dik-dik",
        "Bushpig",
        "Grant's gazelle",
        "Thomson's gazelle",
        "Common ostrich",
        "Aardvark",
        "Cheetah",
        "White-bellied bustard",
        "Wildcat",
        "East African oryx",
        "Buff-crested bustard",
        "Kori bustard",
        "Black-bellied bustard",
        "Great argus",
        "Leopard cat",
        "Banded palm civet",
        "Indian muntjac",
        "Wild boar",
        "Sun bear",
        "Sambar deer",
        "Malayan porcupine",
        "Asian golden cat",
        "Tiger",
        "Three-striped ground squirrel",
        "Common emerald dove",
        "Cape genet",
        "Crested porcupine",
        "African wild dog",
        "Rock hyrax",
        "Dog",
        "unknown bird",
        "unknown bat",
        "Amazonian motmot",
        "Black agouti",
        "Geotrygon",
        "White-nosed coati",
        "Northern tamandua",
        "opossum",
        "crested guan",
        "hummingbird",
        "plain parakeet",
        "dromedary",
        "bat-eared fox",
        "vulturine guineafowl",
        "zebra",
        "aardwolf",
        "serval",
        "greater kudu",
        "hippopotamus",
        "spotted thick-knee",
        "masked palm civet",
        "marbled cat",
        "dhole",
        "asian water monitor",
        "yellow-throated marten",
        "banded linsang",
        "crested partridge",
        "salvadori's pheasant",
        "bronze-tailed peacock-pheasant",
        "sunda pangolin",
        "sumatran serow",
        "macaque",
        "handsome spurfowl",
        "L'Hoest's monkey",
        "black-fronted duiker",
        "african brush-tailed porcupine",
        "chimpanzee",
        "blue monkey",
        "mountain squirrel",
        "western yellow wagtail",
        "yellow-whiskered greenbul",
        "little greenbul",
        "charming thicket rat",
        "forest giant squirrel",
        "Boehm's bush squirrel",
        "yellow-backed duiker",
        "common rufous-nosed rat",
        "moustached grass warbler",
        "Peters's striped mouse",
        "african wading rat",
        "stella wood mouse",
        "servaline genet",
        "side-striped jackal",
        "african pygmy mouse",
        "Ross's turaco",
        "tambourine dove",
        "tullberg's soft-furred mouse",
        "big-eared swamp rat",
        "egyptian goose",
        "link rat",
        "grey-winged francolin",
        "olive thrush",
        "brocket deer",
        "gray fox",
        "ocellated turkey",
        "great curassow",
        "lowland paca",
        "Baird's tapir",
        "raccoon",
        "white-tailed deer",
        "grey-headed dove",
        "central American red brocket",
        "striped hog-nosed skunk",
        "yucatan brown brocket",
        "plain chachalaca",
        "Thomas's langur",
        "sunda clouded leopard",
        "sumatran hog badger",
        "mouse-deer",
        "sumatran treepie",
        "rufous-vented niltava",
        "silver-eared mesia",
        "red-billed partridge",
        "gallopheasants",
        "javan whistling thrush",
        "crestless fireback",
        "crested serpent eagle",
        "blue whistling thrush",
        "collared mongoose",
        "crab-eating fox",
        "motorcycle",
        "deer mouse",
        "jaguarundi",
        "bare-throated tiger heron",
        "blue ground dove",
        "red squirrel",
        "unknown bird",
        "grey-cowled wood rail",
        "unknown dove",
        "central American red brocket",
    ],
}

loader_funcs = {
    "resnet50": get_loaders_main_224,
    "resnet50_pretrained_finetune": get_loaders_main_224,
    "resnet50_pretrained_linearprob": get_loaders_main_224,
    "resnet50_clip_finetune": get_loaders_CLIP,
    "resnet50_clip_linearprob": get_loaders_CLIP,
    "resnet50_clip": get_loaders_CLIP,
    "vitb32": get_loaders_main_224,
    "vitb32_pretrained_finetune": get_loaders_main_224,
    "vitb32_pretrained_linearprob": get_loaders_main_224,
    "vitb32_clip_finetune": get_loaders_CLIP,
    "vitb32_clip_linearprob": get_loaders_CLIP,
    "vitb32_clip": get_loaders_CLIP,
}

_text_archs = [
    "resnet50_clip",
    "resnet50_clip_finetune",
    "vitb32_clip",
    "vitb32_clip_finetune",
]


def get_data(
    dataset_name: str,
    architecture: str,
    batch_size: int,
    num_workers: int,
    root_dir: str = "./data",
    percentage_train: float = 1.0,
    rank: int = 0,
    world_size: int = 1,
):
    """Get the pytorch data loaders based on the architecture"""

    if architecture not in loader_funcs.keys():
        sep = "\n  - "
        valid = sep + sep.join(loader_funcs.keys())
        raise ValueError(
            f"Architecture '{architecture}' not supported. Valid options are: {valid}"
        )

    return DGData(
        dataset_name,
        batch_size,
        num_workers,
        loader_funcs[architecture],
        percentage_train,
        architecture in _text_archs,
        root_dir=root_dir,
        rank=rank,
        world_size=world_size,
    )


class DGData:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int,
        loader: Callable,
        percentage_train: Optional[float] = 1.0,
        text_input: Optional[bool] = False,
        root_dir: Optional[str] = "./data",
        rank: int = 0,
        world_size: int = 1,
    ):
        if (not "camelyon17" in dataset_name) and (not "fmow" in dataset_name) and (not "iwildcam" in dataset_name):
            raise ValueError(f"Dataset {dataset_name} not supported")

        # set local properties
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_input = text_input

        # create loaders
        (
            self.train_loader,
            self.id_val_loader,
            self.id_test_loader,
            self.val_loader,
            self.test_loader,
        ), self.grouper = loader(dataset_name, b_size=batch_size, nw=num_workers, root_dir=root_dir, percentage_train=percentage_train, rank=rank, world_size=world_size)

        # setup text inputs
        self.text_list = None
        if self.text_input:
            name = self.dataset_name.lower().replace("_0", "")
            self.text_list = [text_prefix[name] + x for x in text_suffix[name]]

        # dataset-specific parameters
        self.n_classes = self.train_loader.dataset._n_classes

        if "_0" in dataset_name:
            self.group_id = 0
        else:
            self.group_id = None
            self.grouper = None

    def __str__(self):
        s = ""
        s += "Domain Generalization Data:\n"
        s += f"  name: {self.dataset_name}\n"
        s += f"  batch_size: {self.batch_size}\n"
        s += f"  num_workers: {self.num_workers}\n"
        s += f"  text_input: {self.text_input}\n"
        s += f"  # train batches: {len(self.train_loader)}\n"
        s += f"  # ID val batches: {len(self.id_val_loader)}\n"
        s += f"  # ID test batches: {len(self.id_test_loader)}\n"
        s += f"  # OOD val batches: {len(self.val_loader)}\n"
        s += f"  # OOD test batches: {len(self.test_loader)}\n"

        return s

    def __repr__(self):
        return self.__str__()


def convert_arch_to_filename(arch: str):
    # This is a silly function, but all the preceding training runs used "RN50" in
    # the save filenames instead of "resnet50" (and similar for ViTB32), so this
    # maintains that structure
    s = arch.replace("resnet50", "RN50")
    s = s.replace("vitb32", "ViTB32")
    return s