import torch
from torchvision import transforms
from PIL import Image, ImageTk
import logging
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import time
from networks.models import DenseNet161

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Danh sách mô hình và lớp
MODELS = {
    "Model CD": {
        "path": r"C:\Users\hieuo\OneDrive\Desktop\nam4_ki1\DoAnCN2\LeVanHieu-21AD021\model\semi_CD\epoch_100.pth",
        "class_names": ['ascus', 'asch', 'lsil', 'hsil', 'scc', 'agc', 'trichomonas', 'candida', 'flora', 'herps', 'actinomyces']
    },
    "Model SP": {
        "path": r"C:\Users\hieuo\OneDrive\Desktop\nam4_ki1\DoAnCN2\LeVanHieu-21AD021\model\semi_densenet161\epoch_100.pth",
        "class_names": ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
    },
    "Model LBC": {
        "path": r"C:\Users\hieuo\OneDrive\Desktop\nam4_ki1\DoAnCN2\LeVanHieu-21AD021\model\semi_LBC\epoch_100.pth",
        "class_names": ['Squamous', 'Negative', 'Low', 'High']
    }
}

# Tạo mô hình
current_model = None
current_class_names = None

def create_model(ema=False):
    net = DenseNet161(out_size=len(current_class_names), mode='U-Ones', drop_rate=0.2)
    if len('0,1,2'.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    model = net
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def load_model(model_name):
    global current_model, current_class_names
    model_info = MODELS[model_name]
    model_path = model_info["path"]
    current_class_names = model_info["class_names"]
    logging.info(f"Loading model: {model_name}, Classes: {current_class_names}")

    if not os.path.isfile(model_path):
        messagebox.showerror("Lỗi", f"Không tìm thấy checkpoint tại '{model_path}'")
        return

    model = create_model()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    logging.info(f"Checkpoint keys: {list(checkpoint['state_dict'].keys())}")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    current_model = model

# Định nghĩa các phép biến đổi cho ảnh
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logging.error(f"Lỗi khi tải ảnh: {e}")
        return None, None
    
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        activations, output = current_model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    return predicted_classes.item(), probabilities

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        thread = threading.Thread(target=run_prediction, args=(file_path,))
        thread.start()

def run_prediction(file_path):
    if current_model is None:
        messagebox.showerror("Lỗi", "Vui lòng chọn một mô hình trước!")
        return

    start_time = time.time()  # Thời gian bắt đầu
    predicted_class, probabilities = predict_image(file_path)
    elapsed_time = time.time() - start_time  # Tính thời gian dự đoán

    if predicted_class is not None:
        display_result(predicted_class, probabilities, elapsed_time)
        show_image(file_path)
    else:
        result_label.config(text="Lỗi trong quá trình dự đoán. Vui lòng thử lại.")

def display_result(predicted_class, probabilities, elapsed_time):
    result_text = f'Lớp dự đoán: {current_class_names[predicted_class]}\n'
    result_text += f'Độ chính xác: {probabilities.max().item():.4f}\n'
    result_text += f'Thời gian dự đoán: {elapsed_time:.2f} giây\n'
    
    # Cập nhật nhãn hiển thị kết quả
    result_label.config(text=result_text)

    # Clear the previous data in the table
    for item in tree.get_children():
        tree.delete(item)

    # Insert new data into the table
    for i in range(len(probabilities[0])):
        tree.insert('', 'end', values=(current_class_names[i], f'{probabilities[0][i].item():.4f}'))


def show_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((200, 155))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

def clear_results():
    result_label.config(text="")
    image_label.config(image='')
    image_label.image = None
    
    # Clear the table
    for item in tree.get_children():
        tree.delete(item)

def close_app():
    if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát ứng dụng?"):
        app.destroy()

def on_model_select(event):
    model_name = model_combobox.get()
    load_model(model_name)

# Tạo giao diện ứng dụng
app = tk.Tk()
app.title("ỨNG DỤNG PHÂN LOẠI HÌNH ẢNH Y TẾ")
app.geometry("800x600")
app.configure(bg="#f0f0f0")

# Thanh tiêu đề
title_label = ttk.Label(app, text="Ứng Dụng Phân Loại Ảnh Y Tế", font=("Helvetica", 16), background="#f0f0f0")
title_label.pack(pady=10)

# Tạo khung cho giao diện
frame = ttk.Frame(app, padding=10)
frame.pack(fill='both', expand=True)

# Khung cho hình ảnh
image_frame = ttk.Frame(frame, padding=10, relief='groove', borderwidth=4, style='TFrame')
image_frame.pack(side='left', fill='both', expand=True)

# Khung cho bảng
table_frame = ttk.Frame(frame, padding=10, relief='solid', borderwidth=2, style='TFrame')
table_frame.pack(side='right', fill='y', expand=False)

# Khung chọn mô hình
model_frame = ttk.Frame(app)
model_frame.pack(pady=10)

model_label = ttk.Label(model_frame, text="Chọn mô hình:", background="#f0f0f0")
model_label.pack(side='left', padx=5)

model_combobox = ttk.Combobox(model_frame, values=list(MODELS.keys()), state="readonly")
model_combobox.pack(side='left', padx=5)
model_combobox.bind("<<ComboboxSelected>>", on_model_select)

# Khung cho các nút
button_frame = ttk.Frame(image_frame)
button_frame.pack(pady=20)

# Nút tải ảnh
upload_btn = ttk.Button(button_frame, text="Tải ảnh lên", command=upload_image)
upload_btn.pack(side='left', padx=5)

# Nút xóa kết quả
clear_btn = ttk.Button(button_frame, text="Xóa kết quả", command=clear_results)
clear_btn.pack(side='left', padx=5)

# Nhãn hiển thị hình ảnh
image_label = ttk.Label(image_frame, background="#f0f0f0")
image_label.pack(pady=10)

# Nhãn hiển thị kết quả
result_label = ttk.Label(image_frame, text="", justify='left', background="#f0f0f0", font=("Helvetica", 12))
result_label.pack(pady=10)

# Tạo bảng hiển thị xác suất
tree = ttk.Treeview(table_frame, columns=("Class", "Probability"), show='headings', height=10)
tree.column("Class", width=150, anchor='center')
tree.column("Probability", width=100, anchor='center')
tree.heading("Class", text="Lớp")
tree.heading("Probability", text="Xác suất")     

# Tạo thanh cuộn cho bảng
scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side='right', fill='y')

tree.pack(pady=20, fill='both', expand=True)

# Tạo nút thoát
exit_btn = ttk.Button(app, text="Thoát", command=close_app)
exit_btn.pack(pady=10)

app.mainloop()