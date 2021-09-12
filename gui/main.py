from tkinter import *
from tkinter.ttk import Treeview

import PIL
from PIL import ImageTk, Image
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

mol_width = 400
mol_height = 400

images_reference_list = []
root = Tk()
root.geometry("500x500")
root.title('Retrosynthesis with Transformer')

w = Label(root, text="Please input your molecule as a SMILES string.")
w.pack()

img_label = Label(root, width=mol_width, height=mol_height)

def display(sv):
    # 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
    images_reference_list.clear()
    try:
        smiles = sv.get()
        # Use rdkit to display the molecule
        ibu = Chem.MolFromSmiles(smiles)
        # Create a pillow image
        img = rdkit.Chem.Draw.MolToImage(ibu, size=(mol_width, mol_height))
        # Pillow images need to be transferred to PhotoImage objects
        image_tk = ImageTk.PhotoImage(img)
        images_reference_list.append(image_tk)
        img_label.configure(image=image_tk)
        img_label.image = image_tk
    except ValueError:
        # Nothing to show!
        img_label.config(text='test')
        pass


    print('test')


sv = StringVar()
sv.trace("w", lambda name, index, mode, sv=sv: display(sv))

e = Entry(root, textvariable=sv)
e.pack()

img_label.pack(side="bottom", fill="both", expand="yes")

tree = Treeview(root)
tree.insert("", "end", "A", text="A")
tree.insert("", "end", "B", text="B")
tree.insert("A", "end", "A.1", text="A.1")
tree.insert("A.1", "end", "A.1.1", text="A.1.1")
tree.insert("A", "end", "A.2", text="A.2")
tree.insert("A.2", "end", "A.2.1", text="A.2.1")
tree.insert("A.2", "end", "A.2.2", text="A.2.2")
tree.insert("B", "end", "B.1", text="B.1")
tree.insert("B", "end", "B.2", text="B.2")
tree.insert("B.1", "end", "B.1.1", text="B.1.1")
tree.pack()

root.mainloop()


