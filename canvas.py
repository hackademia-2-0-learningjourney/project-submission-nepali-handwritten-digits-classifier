from tkinter import *
from PIL import Image, ImageDraw, ImageOps

app = Tk()
app.geometry("320x350")  
canvas_size = 320  
output_size = 32   

frame = Frame(app)
frame.pack(fill=BOTH, expand=1)

canvas = Canvas(frame, bg='black', width=canvas_size, height=canvas_size)
canvas.pack(side=TOP, fill=BOTH, expand=1)

image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_some(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), fill='white', width=30)
    draw.line([lasx, lasy, event.x, event.y], fill='white', width=30)
    lasx, lasy = event.x, event.y

def save_canvas_image():
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image_resized = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    # Save the grayscale image
    image_resized.save("canvas_image.png")
    # Close the Tkinter application
    app.destroy()

button_frame = Frame(app)
button_frame.pack(side=BOTTOM, fill=X)

save_btn = Button(button_frame, text="Done", command=save_canvas_image)
save_btn.pack()

canvas.bind('<Button-1>', get_x_and_y)
canvas.bind('<B1-Motion>', draw_some)

app.mainloop()
