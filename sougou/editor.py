# -*- coding=utf-8 -*-
from Tkinter import *
from tkSimpleDialog import *
from tkFileDialog import *
from tkMessageBox import *
import os

class QuitMe(Frame):                        
    def __init__(self, parent=None):          
        Frame.__init__(self, parent)
        self.pack()
        widget = Button(self, text='退出', command=self.quit)
        widget.pack(expand=YES, fill=BOTH, side=LEFT)

    def quit(self):
        ans = askokcancel('退出', "真的要退出吗？")
        if ans:
            Frame.quit(self)


class ScrolledText(Frame):
    def __init__(self, parent=None, text='', file_name=None):
        Frame.__init__(self, parent)
        self.pack(expand=YES, fill=BOTH)               
        self.makewidgets()
        self.file_name = file_name
        self.settext(text)

    def makewidgets(self):
        sbar = Scrollbar(self)
        text = Text(self, relief=SUNKEN)
        sbar.config(command=text.yview)                  
        text.config(yscrollcommand=sbar.set)           
        sbar.pack(side=RIGHT, fill=Y)                   
        text.pack(side=LEFT, expand=YES, fill=BOTH)     
        self.text = text

    def settext(self, text=''):
        if self.file_name:
            try:
                f = open(self.file_name, 'r')
                text = f.read()
                f.close()
            except IOError:
                open(self.file_name, 'w')
                text = ''
        self.text.delete('1.0', END)                   
        self.text.insert('1.0', text)                  
        self.text.mark_set(INSERT, '1.0')              
        self.text.focus()

    def gettext(self):                               
        return self.text.get('1.0', END+'-1c')         


class SimpleEditor(ScrolledText):                        
    def __init__(self, parent=None, file_name=None):
        frm = Frame(parent)
        frm.pack(fill=X)
        Button(frm, text='保存',  command=self.onSave).pack(side=LEFT)
        Button(frm, text='剪切',   command=self.onCut).pack(side=LEFT)
        Button(frm, text='粘贴', command=self.onPaste).pack(side=LEFT)
        Button(frm, text='查找',  command=self.onFind).pack(side=LEFT)
        QuitMe(frm).pack(side=LEFT)
        ScrolledText.__init__(self, parent, file_name=file_name)
        self.text.config(font=('courier', 9, 'normal'))

    def onSave(self):
        if self.file_name:
            alltext = self.gettext()
            open(self.file_name, 'w').write(alltext.encode('utf-8'))
        else:
            filename = asksaveasfilename()
            if filename:
                alltext = self.gettext()
                open(filename, 'w').write(alltext)

    def onCut(self):
        text = self.text.get(SEL_FIRST, SEL_LAST)        
        self.text.delete(SEL_FIRST, SEL_LAST)           
        self.clipboard_clear()              
        self.clipboard_append(text)

    def onPaste(self):                                    
        try:
            text = self.selection_get(selection='CLIPBOARD')
            self.text.insert(INSERT, text)
        except TclError:
            pass

    def onFind(self):
        target = askstring('SimpleEditor', '搜索')
        if target:
            where = self.text.search(target, INSERT, END)  
            if where:                                    
                print where
                pastit = where + ('+%dc' % len(target))   
                # self.text.tag_remove(SEL, '1.0', END)
                self.text.tag_add(SEL, where, pastit)     
                self.text.mark_set(INSERT, pastit)         
                self.text.see(INSERT)                    
                self.text.focus()
# if there are no cmdline arguments, open a new file.
if len(sys.argv) > 1:
    import os
    print(os.getcwd())
    SimpleEditor(file_name=sys.argv[1]).mainloop()
else:
    SimpleEditor().mainloop()
