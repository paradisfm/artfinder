#program that analyzes color scheme of submitted picture
#input - budget & multiselect menu for styles 
#finds 5 options from etsy for wall art
import model.colors as colors
import model.objects as objects
#get picture
#file = easygui.fileopenbox(msg="Hello! What does the space you'd like to decorate look like?")
#img = skimage.io.imread(fname=file)

#TODO: input from user

#TODO: the entire front

if __name__ == "__main__":
    df = colors.get_rgb(objects.img)
    k = colors.get_color_nums(df)
    colorscheme = colors.get_colors(df, k)





