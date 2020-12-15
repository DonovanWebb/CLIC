import numpy as np
import matplotlib.pyplot as plt


def update_annot(ind, sc, annot):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([str(names[n]) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    if event.inaxes == ax1:
        vis = annot1.get_visible()
        cont, ind = sc1.contains(event)
        if cont:
            update_annot(ind, sc1, annot1)
            annot1.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot1.set_visible(False)
                fig.canvas.draw_idle()


def plot(sinos, num, l_size):
    # splitting into classes

    # sinos1 = np.array([sinos[i] for i in range(sinos.shape[0])
    #                    if np.floor(i/l_size) % 4 == 0])
    # sinos2 = np.array([sinos[i] for i in range(sinos.shape[0])
    #                    if np.floor(i/l_size) % 4 == 1])
    # sinos3 = np.array([sinos[i] for i in range(sinos.shape[0])
    #                    if np.floor(i/l_size) % 4 == 2])
    # sinos4 = np.array([sinos[i] for i in range(sinos.shape[0])
    #                    if np.floor(i/l_size) % 4 == 3])

    sinos1 = np.array([sinos[i] for i in range(sinos.shape[0])
                       if np.floor(i/l_size) % 2 == 0])
    sinos2 = np.array([sinos[i] for i in range(sinos.shape[0])
                       if np.floor(i/l_size) % 2 == 1])
    plt.figure(12)
    plt.scatter(sinos[:, 0],
                      sinos[:, 1],
                      s=5,
                      color='r')
    i = 0
    plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1], s = 5, color='yellow')
    i = 1
    plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1], s = 5, color='brown')
    plt.axis('off')
    #plt.savefig('./figures/TSNE_raw', dpi = 1080, pad_inches=0)


    global fig
    fig = plt.figure(figsize=(8, 8))
    global ax1
    ax1 = fig.add_subplot()
    global sc1
    sc1 = ax1.scatter(sinos[:, 0],
                      sinos[:, 1],
                      s=5,
                      color='r')
    for it in range(90,91):
        i = it - 90
        ax1.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1], s = 5, color='g')
    # ax1.grid()
    global annot1
    annot1 = ax1.annotate("", xy=(0, 0), xytext=(20, 20),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)
    global names
    names = np.array(list(range(0, num*l_size)))
    # names = np.array([(np.floor(x/l_size),np.floor(x/l_size)%4) for x in names])
    names = np.array([(np.floor(x/l_size), np.floor(x/l_size) % 2) for x in names])
    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Classes
    plt.figure(291)
    plt.scatter(sinos1[:, 0], sinos1[:, 1], alpha=0.6, color='r', s = 5)
    plt.scatter(sinos2[:, 0], sinos2[:, 1], alpha=0.6, color='b', s = 5)
    # plt.scatter(sinos3[:, 0], sinos3[:, 1], alpha=0.6, color='g', s = 5)
    # plt.scatter(sinos4[:, 0], sinos4[:, 1], alpha=0.6, color='orange', s = 5)
    plt.axis('off')
    #plt.savefig('./figures/TSNE_gt', dpi = 1080, pad_inches=0)

    '''
    # Individual sinograms
    plt.figure(3183)
    for i in range(num):
        plt.scatter(sinos[l_size*i:l_size*(i+1), 0], sinos[l_size*i:l_size*(i+1), 1], alpha=0.6, s = 5)

    plt.figure(90)
    for it in range(90,95):
        i = it - 90
        plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1],alpha=0.6)
    plt.axis('off')
    #plt.savefig('./figures/10_sin', dpi = 1080, pad_inches=0)

    # plotting found clusters
    plt.figure(31)
    results = [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1,21, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1,21, 1,40, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0,55, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,93, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    for i in range(num):
        if results[i] == 0:
            plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1],alpha=0.6,color='r', s = 5)
        elif results[i] == 1:
            plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1],alpha=0.6,color='b', s = 5)
        else:
            plt.scatter(sinos[l_size*i:l_size*(i+1),0],sinos[l_size*i:l_size*(i+1),1],alpha=0.6,color='g', s = 5)
    plt.axis('off')
    # plt.savefig('./figures/TSNE_exp', dpi = 1080, pad_inches=0)
    '''
