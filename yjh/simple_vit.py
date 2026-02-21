import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import os
import logging
import json


# 设置日志记录配置
logging.basicConfig(
    filename='training_log.txt',   # 设置日志文件名
    level=logging.INFO,            # 设置日志级别为 INFO，包含 INFO 及更高的日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 记录开始的训练信息
logging.info("Training started.")
def load_filenames_from_json(input_file):
    with open(input_file, 'r') as f:
        filenames = json.load(f)
    return filenames
filenames_test=[]
# 从文件中读取文件名
filenames_train = load_filenames_from_json('/home/yjh/code_yjh_bishe/train_filenames_list.json')
wrong_filenames = load_filenames_from_json('/home/yjh/code_yjh_bishe/wrong_names.json')
for i in wrong_filenames:
    filenames_train.remove(i)
for i in os.listdir("/data1/yuanjiahong_files/bishe/EndoVis2018/data_depthmap_test"):
    filenames_test.append(i.split('.')[0])
class dataset_vit(torch.utils.data.Dataset):
    def __init__(self, data_path,ori_path,version, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.ori_path=ori_path
        self.version=version
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, index):
        if self.version == 0:
            path = self.ori_path+f"{self.data_path[index]}.png"
        else:
            a,b=self.data_path[index].split('-')
            path = self.ori_path+a+f"_{self.version}"+"-"+b+".png"

            
        image = Image.open(path)
            # 确保图像是灰度图像，然后将其转换为伪 RGB 图像
        image = image.convert("L")  # 'L' 模式是灰度图
        
        if self.transform:
            image = self.transform(image)
        label = int(self.data_path[index].split('-')[-1].split('.')[0])-1 #0~6
        return image, label
                
        
# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes=7

# from transformers import AutoImageProcessor, AutoModelForImageClassification

# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
# 使用 COCO 预训练的 ViT 模型，加载预训练权重
model_name = "google/vit-base-patch16-224-in21k"  # COCO 上预训练的 ViT 模型
model = ViTForImageClassification.from_pretrained(model_name, num_labels=7).to(device)

# 加载 ViT 模型对应的 Feature Extractor（图像预处理）
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 的输入大小通常是 224x224
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5]),  # COCO 数据集的均值和标准差
])

# 加载 CIFAR-10 数据集或其他你的数据集（可以换成 COCO 数据集）
# 这里为了示例，我们使用 CIFAR-10 数据集


# 分割成训练和验证数据集




# 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=4e-5)
criterion = nn.CrossEntropyLoss()

# 训练模型
n_epochs = 100

def train_one_epoch(model, dataloader, optimizer, criterion, num_classes, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        # 提取特征并进行前向传播
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 获取预测类别
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 存储所有真实标签和预测概率
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())  # 获取概率分布

    # 计算准确率
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)

    # 计算 AUC（one-vs-rest 方式）
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr', labels=[i for i in range(num_classes)])

    return avg_loss, accuracy, auc


def evaluate(model, dataloader, criterion, num_classes, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # 用于存储每个类别的正确数量和总数量
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 存储真实标签和预测概率
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            # 更新每个类别的正确预测数量和总数量
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # 计算总体准确率
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)

    # 计算 AUC（one-vs-rest 方式）
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr', labels=[i for i in range(num_classes)])

    # 计算每个类别的准确率
    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

    return avg_loss, accuracy, auc, class_accuracy

# 训练和验证循环
best_auc = 0.0
for epoch in range(1, n_epochs):
    if epoch % 2 == 0  :
        version = 0 
    else:
        version = int((epoch % 32 + 1)/2)
    train_dataset = dataset_vit(filenames_train,"/data1/yuanjiahong_files/bishe/EndoVis2018/data_depthmap_train/",version,transform=transform)
    test_dataset = dataset_vit(filenames_test,"/data1/yuanjiahong_files/bishe/EndoVis2018/data_depthmap_test/",0, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Epoch {epoch+1}/{n_epochs}")
    logging.info(f"Epoch {epoch+1}/{n_epochs}")
    # 训练阶段
    train_loss, train_accuracy,train_auc = train_one_epoch(model, train_loader, optimizer, criterion, num_classes, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}", f"Train AUC: {train_auc:.4f}")
    logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}")
    # 验证阶段
    val_loss, val_accuracy, val_auc,class_accuracy = evaluate(model, test_loader, criterion, num_classes, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
    logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.4f}")
    # 保存最优模型权重
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "/data1/yuanjiahong_files/bishe/EndoVis2018/vit/weight/best_vit_7_class_model_{:.4f}.pth".format(val_auc))
        print("Saved best model weights.")


#  /home/yjh/anaconda3/envs/yjh/bin/python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/simple_vit.py 