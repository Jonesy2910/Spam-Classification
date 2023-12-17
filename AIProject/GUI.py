import tkinter as tk


def input(input_text, model, tv):
    input_features = tv.transform([input_text])
    predict = model['Stack'].predict(input_features)
    return predict[0]


def button_click(entry, result_label, model, tv):
    input_text = entry.get()
    predict = input(input_text, model, tv)
    if predict == 1:
        result = "This is SPAM. Do not use the email."
        colour = "red"
        image = tk.PhotoImage(file='rounded_red.png')
    else:
        result = "This is NOT SPAM. You can use this email."
        colour = "green"
        image = tk.PhotoImage(file='rounded_green.png')
    result_label.config(text=result, foreground=colour)
    result_label.configure(image=image)
    result_label.image = image


def run_gui(model, tv):
    window = tk.Tk()
    window.title("Spam Classifier")
    window.geometry("500x450")

    title_label = tk.Label(window, text="Spam Classifier", font=("MS Gothic", 18, "bold", "underline"))
    title_label.pack(pady=20)

    input_frame = tk.Frame(window)
    input_frame.pack()

    label = tk.Label(input_frame, text="Enter E-Mail Here:", font=("MS Gothic", 12, "underline"))
    label.pack(side='left', padx=10, pady=10)

    entry_image = tk.PhotoImage(file='rounded.png')
    entry_img = tk.Label(input_frame, image=entry_image)
    entry_img.pack(side='left', padx=10, pady=10)

    input_text = tk.StringVar()
    entry = tk.Entry(entry_img, textvariable=input_text, border=0, font=("MS Gothic", 12))
    entry.place(relx=0.5, rely=0.5, anchor='center')

    button_image = tk.PhotoImage(file='button.png')
    button = tk.Button(window, image=button_image, border=0, command=lambda: button_click(entry, result_label, model, tv))
    button.pack(pady=20)

    result_image = tk.PhotoImage(file='rounded.png')
    result_label = tk.Label(window, text="", font=("MS Gothic", 7), image=result_image, compound="center")
    result_label.pack()
    image = tk.PhotoImage(file='Image.png')
    image_label = tk.Label(window, image=image)
    image_label.pack()

    window.mainloop()
