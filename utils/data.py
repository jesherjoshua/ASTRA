import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from collections import Counter


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        
class iCIFAR224_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.01, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")
class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/imagenet-r/train/"
        test_dir = "../data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class iImageNetR_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.1, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dir = "../data/imagenet-r/train/"
        test_dir = "../data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")

class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/imagenet-a/train/"
        test_dir = "../data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/cub/train/"
        test_dir = "../data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class MedMNIST(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/kaggle/input/medmnist-merged-for-continual-learning/train/"
        test_dir = "/kaggle/input/medmnist-merged-for-continual-learning/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CUB_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.25, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dir = "../data/cub/train/"
        test_dir = "../data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")



class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/objectnet/train/"
        test_dir = "../data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class objectnet_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.25, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dir = "../data/objectnet/train/"
        test_dir = "../data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes // 4  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")



class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/omnibenchmark/train/"
        test_dir = "../data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.25, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(300).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dir = "../data/imagenet-r/train/"
        test_dir = "../data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes // 4  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")

class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/vtab/train/"
        test_dir = "../data/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab_imbalanced(iData):
    def __init__(self, args, imbalance_ratio=0.1, imbalance_classes=None):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(50).tolist()
        self.imbalance_ratio = imbalance_ratio
        self.imbalance_classes = imbalance_classes

    def download_data(self):
        train_dir = "../data/vtab/train/"
        test_dir = "../data/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # Apply class imbalance to the training data
        self.apply_class_imbalance()

    def apply_class_imbalance(self): 
        np.random.seed(1993)
        if self.imbalance_classes is None:
            # Select a subset of classes to imbalance
            num_classes = len(set(self.train_targets))
            num_imbalance_classes = num_classes // 2  # Example: Imbalance 25% of classes
            self.imbalance_classes = np.random.choice(
                np.arange(num_classes), num_imbalance_classes, replace=False
            )
    
        # Create new training data with imbalance
        new_train_data = []
        new_train_targets = []
    
        class_counts = Counter(self.train_targets)
        for cls in class_counts.keys():
            cls_indices = np.where(np.array(self.train_targets) == cls)[0]
            if cls in self.imbalance_classes:
                # Reduce the number of samples for this class
                reduced_indices = np.random.choice(
                    cls_indices, int(len(cls_indices) * self.imbalance_ratio), replace=False
                )
                new_train_data.extend(np.array(self.train_data)[reduced_indices])
                new_train_targets.extend(np.array(self.train_targets)[reduced_indices])
            else:
                # Keep all samples for other classes
                new_train_data.extend(np.array(self.train_data)[cls_indices])
                new_train_targets.extend(np.array(self.train_targets)[cls_indices])
    
        # Ensure train_data and train_targets are NumPy arrays
        self.train_data = np.array(new_train_data)
        self.train_targets = np.array(new_train_targets)
    
        # Count and print the number of samples per class after applying imbalance
        new_class_counts = Counter(self.train_targets)
        print("\nNumber of samples per class in the training set AFTER imbalance:")
        for cls, count in new_class_counts.items():
            print(f"Class {cls}: {count} samples")
