import argparse, os, traceback
import tkinter as tk
import torch
import colorize
import platform
from upscale import upscale
from PIL import ImageTk, Image, ImageOps
from tkinter.filedialog import askopenfilename, asksaveasfilename

CUDA_AVAILABLE = torch.cuda.device_count() > 0

def get_resized_img(pil_img, max_pixel=512, resample=Image.ANTIALIAS):
    size = pil_img.size
    w, h = size[0], size[1]
    if w > h:
        rs_size = (max_pixel, int(max_pixel / w * h))
    else:
        rs_size =  (int(max_pixel / h * w), max_pixel)
    return pil_img.resize(rs_size, resample=resample)

def get_tagset():
    tag_list = []
    with open('loader/tags.txt', 'r') as f:
        for line in f:
            tag_list.append(line.strip())
    return tag_list

class ChoiceBox(tk.Frame):
    def __init__(self, list_items, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.list_items = list_items
        self.list_var = tk.StringVar()
        self.list_var.set(' '.join(list_items))
        self.scroll = tk.Scrollbar(self)
        self.list_box = tk.Listbox(self, listvariable=self.list_var, selectmode=tk.MULTIPLE, 
            height=30)

        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_box.pack()

        self.list_box.config(yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.list_box.yview)

    def get_selected(self):
        return list(map(lambda i: self.list_items[i], 
            self.list_box.curselection()))


class App(object):
    def __init__(self):
        self.top = tk.Tk()
        self.top.title = 'tag2pix'
        self.img_panel = tk.Frame(self.top, height=512, width=512)
        self.img_panel.pack_propagate(0)
        self.img_label = tk.Label(self.img_panel)
        self.stat_text = tk.Text(self.top, height=6, relief='sunken', borderwidth=1)
        self.tag_list = get_tagset()
        self.choice_box = ChoiceBox(self.tag_list, self.top)

        self.sketch_img = None
        self.current_img = None
        self.upscale_sizes = [32, 64, 128, 256, 512, 768, 1024]
        self.upscale_var = tk.IntVar()
        self.use_gpu = tk.BooleanVar()
        self.use_512 = tk.BooleanVar()

        self.btn_load = tk.Button(self.top, text='Load Sketch', command=self.load_file)
        self.btn_colorize = tk.Button(self.top, text='Colorize', command=self.colorize_sketch)
        self.btn_upscale = tk.Button(self.top, text='Upscale', command=self.upscale_img)
        self.btn_save = tk.Button(self.top, text='Save', command=self.save_file)
        self.cb_use512 = tk.Checkbutton(self.top, text='512px model', variable=self.use_512)
        self.cb_use512.var = self.use_512
        self.cb_gpu = tk.Checkbutton(self.top, text='GPU | Output Size: ', variable=self.use_gpu)
        self.cb_gpu.var = self.use_gpu
        self.drop_upscale_size = tk.OptionMenu(self.top, self.upscale_var, *self.upscale_sizes)

        self.btn_load.grid(row=0, column=0, sticky="nesw")
        self.btn_colorize.grid(row=0, column=1, sticky="nesw")
        self.btn_upscale.grid(row=0, column=2, sticky="nesw")
        self.btn_save.grid(row=0, column=3, sticky="nesw")
        self.cb_use512.grid(row=0, column=4, sticky="nesw")
        self.cb_gpu.grid(row=0, column=5, sticky="nesw")
        self.drop_upscale_size.grid(row=0, column=6, ipadx=3, sticky="ew")
        self.drop_upscale_size.config(width=5)

        self.stat_text.grid(row=1, column=0, columnspan=7, sticky="nesw")
        
        self.img_panel.grid(row=2, column=0, columnspan=5, sticky="nsw")
        self.img_label.pack()

        self.choice_box.grid(row=2, column=5, columnspan=2, sticky="nse")

        self.upscale_var.set(768)
        self.use_512.set(True)

        self.print_status('Please Load Sketch File')

        if platform.system() != 'Windows':
            self.print_error('Cannot use Upscaling (Windows only)')
            self.btn_upscale.configure(state='disabled')

        if CUDA_AVAILABLE:
            self.use_gpu.set(True)
            torch.backends.cudnn.benchmark = True
            self.print_status('Found CUDA GPU, run in gpu mode')
        else:
            self.use_gpu.set(False)
            self.cb_gpu.configure(state='disabled')
            self.print_error('Failed to find CUDA GPU, run in cpu mode')

        self.use_old.set(False)

    def print_log(self, status):
        self.stat_text.configure(state='normal')
        self.stat_text.insert(tk.END, status + '\n')
        self.stat_text.see(tk.END)
        self.stat_text.configure(state='disabled')
    def print_status(self, status):
        self.print_log(f'Status: {status}')
    def print_error(self, status):
        self.print_log(f'Error: {status}')

    def set_img(self, pil_img):
        self.current_img = pil_img
        resize_img = get_resized_img(pil_img)
        img_tk = ImageTk.PhotoImage(resize_img)
        img_tk.height = 512
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk

    def load_file(self):
        file_name = askopenfilename()
        if os.path.exists(file_name):
            self.sketch_img = Image.open(file_name).convert('RGB')
            self.set_img(self.sketch_img)
            w, h = self.sketch_img.size
            self.print_status(f'Load Sketch File: "{file_name}" ({h}x{w})')
        else:
            self.print_status(f'"{file_name}" does not exist or is not an image file.')

    def save_file(self):
        if self.current_img is None:
            self.print_error('Cannot find current image.')
            return

        file_name = asksaveasfilename()
        if file_name:
            try:
                if file_name[-4:].lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
                    file_name = file_name + '.png'
                if os.path.exists(file_name):
                    pass # TODO
                self.current_img.save(file_name)
                self.print_status(f'Image saved to "{file_name}"')
            except Exception as e:
                traceback.print_exc()
                self.print_error(f'Failed to save image: "{file_name}"')
        else:
            self.print_error('Invalid file name')

    def colorize_sketch(self):
        if self.sketch_img is None:
            self.print_error('Please Load Sketch Image')
            return
        gpu = self.use_gpu.get()
        use_512 = self.use_512.get()
        enable_str = 'enabled' if gpu else 'disabled'
        self.print_status(f'Colorize with {512 if use_512 else 256}px model / GPU {enable_str}')

        target_img = self.sketch_img # ImageOps.autocontrast(self.sketch_img, ignore=255)
        
        try:
            color_img = colorize.colorize(target_img, 
                self.choice_box.get_selected(), 
                gpu=self.use_gpu.get(),
                input_size=512 if use_512 else 256)
            self.set_img(color_img)
            w, h = color_img.size
            self.print_status(f'Colorized result: ({w}x{h})px')
        except Exception as e:
            traceback.print_exc()
            self.print_error('Failed to colorize sketch. Check stack trace')

    def upscale_img(self):
        if self.current_img is None:
            self.print_error('Please Load Image')
        return
        
        gpu = self.use_gpu.get()
        height = self.upscale_var.get()
        self.print_status(f'Upscale current image to {height}px')

        try:
            upscaled_img = upscale(self.current_img, gpu, height)
            w, h = upscaled_img.size
            self.print_status(f'Finished Upscaling: ({w}x{h})')
            self.set_img(upscaled_img)
        except Exception as e:
            traceback.print_exc()
            self.print_error('Failed to upscale sketch. Check stack trace.')

if __name__ == '__main__':
    app = App()
    app.top.mainloop()
