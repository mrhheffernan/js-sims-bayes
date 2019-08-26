#!/usr/bin/env python3
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_tkagg import \
                FigureCanvasTkAgg#, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        self.Names = ['a','b','c']
        self.idf = 0
        super().__init__(master)
        self.createWidgets()

    def toggle(self):
        self.idf = (self.idf + 1)%5
        self.change_idf.config(text='df={:d}, click to change'.format(self.idf))

    def createWidgets(self):
        fig, self.axes =plt.subplots(nrows=2, ncols=2, figsize=(4,4))

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0,column=1)
        plt.tight_layout(True)
        plt.subplots_adjust(top=0.9)
        self.canvas.draw()

        # plot
        self.plotbutton=tk.Button(master=root, text="plot", 
                           command=lambda: self.plot())
        self.plotbutton.grid(row=1,column=0)

        # switching delta f
        self.change_idf = tk.Button(text="df=0, click to switch", 
                 command=lambda: [self.toggle(), self.plot()])
        self.change_idf.grid(row=2, column=0)

        # quit
        self.quit = tk.Button(master=root, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.grid(row=3, column=0)

        for i, name in enumerate(self.Names):
            l = 0.
            h = 1.
            setattr(self, name, l)
            # add the slide scale for this variable
            setattr(self, 'tune'+name, 
                    tk.Scale(
		        master=root, 
		        from_=l, to=h, resolution=(h-l)/30., length=400, 
		        orient="horizontal", tickinterval=1)
                    )

            # labelling the slide scale
            setattr(self, 'label'+name, tk.Label(master=root, text=name))

            getattr(self, 'label'+name).grid(row=i+1, column=2)
            getattr(self, 'tune'+name).set(.5)
            getattr(self, 'tune'+name).grid(row=i+1, column=1, columnspan=2)
            getattr(self, 'tune'+name).bind("<B1-Motion>",
                          lambda event: self.plot() )  

    def formatting_plot(f):
        def ff(self):
            f(self)
            plt.tight_layout(True)
            plt.subplots_adjust(top=0.9)
            self.canvas.draw()
        return ff
     
    @formatting_plot
    def plot(self):
        params = [getattr(self, 'tune'+name).get() for name in self.Names]
        a, b, c = params
        def y(x, i):
            return (a + b*x**c)**(1+i-self.idf)
        x = np.linspace(0,1,100)
        for i, ax in enumerate(self.axes.flatten()):
            if len(ax.lines) >= 1:
                ax.lines[-1].remove()
            ax.plot(x, y(x, i), 'b-')
            ax.set_xlim(0,1)
            ax.set_ylim(0,2)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.master.title('Hand tuning your parameters')
    app.mainloop()
