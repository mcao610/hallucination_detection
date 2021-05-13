import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_scatter(input_data, 
                 labels, 
                 x_label='Prior Probability', 
                 y_label='Posterior Probability',
                 colors=['tab:blue', 'tab:green', 'tab:orange', 'tab:red']):
    assert len(input_data) == len(labels)
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    
    for i, data in enumerate(input_data):
        ax.scatter([d[0] for d in data], 
                   [d[1] for d in data], c=colors[i], s=64, label=labels[i], alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)
    
    # plt.savefig('foo.png')
    plt.show()
    

def plot(taskname, 
         priors, 
         posteriors, 
         labels, 
         n_neighbors=15, 
         fig_size=(4.5, 3.5), 
         colors=['red', 'blue'], 
         legend_labels=['Non-factual', 'Factual'],
         x_label='Prior Probability',
         y_label='Posterior Probability',
         save_figure=False):
    """Draw KNN classification boundaries."""
    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')

    larray = np.array(labels)
    priors = np.array(priors)
    posteriors = np.array(posteriors)
    
    pin = posteriors
    x_mat = np.vstack([pin / np.std(pin), priors / np.std(priors)]).transpose()
    y_vec = np.array(labels)
    
    classifier.fit(x_mat,y_vec)
    
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = x_mat[:, 0].min() - .5, x_mat[:, 0].max() + .5
    y_min, y_max = x_mat[:, 1].min() - .5, x_mat[:, 1].max() + .5
    
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    xx_1, yy_1 = np.meshgrid(np.arange(x_min, x_max, h) * np.std(pin),
                             np.arange(y_min, y_max, h) * np.std(priors))
    
    # Plot color map
    cm = plt.cm.RdBu
    fig, ax = plt.subplots(figsize=fig_size)
    ax.contourf(xx_1, yy_1, Z, cmap=cm, alpha=.7)
    
    ax.scatter(np.array(pin)[np.nonzero(larray==1)[0]], np.array(priors)[np.nonzero(larray==1)[0]], 
               color=colors[1], edgecolor='black', label=legend_labels[1], alpha=0.8)
    ax.scatter(np.array(pin)[np.nonzero(larray==0)[0]], np.array(priors)[np.nonzero(larray==0)[0]], 
               color=colors[0], edgecolor='black', label=legend_labels[0], marker='s', alpha=0.8)

    
    # Plot Hist diagram
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 0.7, pad=0.0, sharex=ax)
    axHisty = divider.append_axes("right", 0.7, pad=0.0, sharey=ax)
    axHistx.xaxis.set_tick_params(labelbottom=False, bottom=False)
    axHistx.yaxis.set_tick_params(labelleft=False, left=False)
    axHisty.xaxis.set_tick_params(labelbottom=False, bottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False, left=False)
    b_subset = np.nonzero(larray==1)[0]
    r_subset = np.nonzero(larray==0)[0]
    x = np.array(pin)
    y = np.array(priors)
    _ = axHistx.hist(x[b_subset], color=colors[1], bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHistx.hist(x[r_subset], color=colors[0], bins=np.arange(x_min, x_max, 0.3)*np.std(pin), alpha=0.7)
    _ = axHisty.hist(y[b_subset], color=colors[1], bins=np.arange(y_min, y_max, 0.3)*np.std(priors), orientation='horizontal', alpha=0.7)
    _ = axHisty.hist(y[r_subset], color=colors[0], bins=np.arange(y_min, y_max, 0.3)*np.std(priors), orientation='horizontal', alpha=0.7)

    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(np.min(xx_1), np.max(xx_1))
    ax.set_ylim(np.min(yy_1), np.max(yy_1))
    ax.set_xlabel(x_label, fontweight ='bold', fontsize=11)
    ax.set_ylabel(y_label, fontweight ='bold', fontsize=11)
#     axHistx.set_title(taskname)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    if save_figure:
        plt.savefig("figures/" + taskname +'.pdf')
    plt.show()


def plot_three_with_boundary(taskname, 
                             priors, 
                             posteriors, 
                             labels, 
                             n_neighbors=15, 
                             fig_size=(4.5, 3.5), 
                             colors=['red', 'blue'], 
                             legend_labels=['Non-factual', 'Factual'],
                             x_label='Prior Probability',
                             y_label='Posterior Probability',
                             save_figure=False,
                             interval=0.2,
                             h=0.02):
    """Draw KNN classification boundaries."""
    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='brute')

    larray = np.array(labels)
    priors = np.array(priors)
    posteriors = np.array(posteriors)
    
    pin = posteriors
    x_mat = np.vstack([pin / np.std(pin), priors / np.std(priors)]).transpose()
    y_vec = np.array(labels)
    
    classifier.fit(x_mat, y_vec)
    
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = x_mat[:, 0].min() - .5, x_mat[:, 0].max() + .5
    y_min, y_max = x_mat[:, 1].min() - .5, x_mat[:, 1].max() + .5
    
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_prediction = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))  # [80, 72, 3]
    Z_prediction = Z_prediction.reshape(xx.shape)

    xx_1, yy_1 = np.meshgrid(np.arange(x_min, x_max, h) * np.std(pin),
                             np.arange(y_min, y_max, h) * np.std(priors))
    
    # Plot color map
    fig, ax = plt.subplots(figsize=fig_size)
    z_max_index = np.argmax(Z, axis=2)
    nb_classes = 3
    one_hot_max_index = np.eye(nb_classes)[z_max_index.reshape(-1)]
    one_hot_max_index = one_hot_max_index.reshape(Z.shape)
    Z_c = Z * one_hot_max_index
    Z_c = Z_c + (one_hot_max_index - 1)

    ax.contourf(xx_1, yy_1, Z_c[:, :, 0], cmap=plt.cm.Blues, alpha=.5, levels=np.arange(0.0, 1.1, interval))
    ax.contourf(xx_1, yy_1, Z_c[:, :, 1], cmap=plt.cm.Greens, alpha=.7, levels=np.arange(0.0, 1.1, interval))
    ax.contourf(xx_1, yy_1, Z_c[:, :, 2], cmap=plt.cm.Reds, alpha=.5, levels=np.arange(0.0, 1.1, interval))

#     cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA'])
#     ax.pcolormesh(xx_1, yy_1, Z_prediction, cmap=cmap_light, alpha=.8)

    ax.scatter(np.array(pin)[np.nonzero(larray==0)[0]], np.array(priors)[np.nonzero(larray==0)[0]], color=colors[0], 
               edgecolor='black', label=legend_labels[0], marker='s', alpha=0.8)
    ax.scatter(np.array(pin)[np.nonzero(larray==1)[0]], np.array(priors)[np.nonzero(larray==1)[0]], color=colors[1], 
               edgecolor='black', label=legend_labels[1], alpha=0.8)
    ax.scatter(np.array(pin)[np.nonzero(larray==2)[0]], np.array(priors)[np.nonzero(larray==2)[0]], color=colors[2], 
               edgecolor='black', label=legend_labels[2], marker="D", alpha=0.8)

    ax.legend(loc='lower right')
    ax.set_xlim(np.min(xx_1), np.max(xx_1))
    ax.set_ylim(np.min(yy_1), np.max(yy_1))
    ax.set_xlabel(x_label, fontweight ='bold', fontsize=11)
    ax.set_ylabel(y_label, fontweight ='bold', fontsize=11)
#     axHistx.set_title(taskname)
    
    plt.tight_layout()
    if save_figure:
        plt.savefig("figures/" + taskname +'.pdf')
    plt.show()


def plot_three(taskname, 
               priors, 
               posteriors, 
               labels, 
               n_neighbors=15, 
               fig_size=(4.5, 3.5), 
               colors=['green', 'blue', 'red'], 
               x_label='CMLM trained on CNN/DM from scratch',
               y_label='CMLM trained on XSum from scratch',
               legend_labels=['Non-hallucinated', 'Factual Hallucinataion', 'Non-factual Hallucinataion'],
               save_figure=False):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    
    priors = np.array(priors)
    larray = np.array(labels)
    posteriors = np.array(posteriors)
    pin = posteriors

    # Plot color map
    cm = plt.cm.RdBu
    fig, ax = plt.subplots(figsize=fig_size)

    ax.scatter(np.array(pin)[np.nonzero(larray==0)[0]], np.array(priors)[np.nonzero(larray==0)[0]], color=colors[0], 
               edgecolor='black', label=legend_labels[0], marker='s', alpha=0.8)
    ax.scatter(np.array(pin)[np.nonzero(larray==1)[0]], np.array(priors)[np.nonzero(larray==1)[0]], color=colors[1], 
               edgecolor='black', label=legend_labels[1], alpha=0.8)
    ax.scatter(np.array(pin)[np.nonzero(larray==2)[0]], np.array(priors)[np.nonzero(larray==2)[0]], color=colors[2], 
               edgecolor='black', label=legend_labels[2], marker="D", alpha=0.8)
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel(x_label, fontweight='bold', fontsize=11)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    if save_figure:
        plt.savefig("figures/" + taskname +'.pdf')
    plt.show()
    
    
def draw_auc(target, probs, labels, colors, fig_name=None):
    assert len(probs) == len(labels) == len(colors)
    plt.figure(figsize=(4.5, 3.5))
    lw = 2
    
    for prob, l, c in zip(probs, labels, colors):
        fpr, tpr, _ = roc_curve(np.asarray(target), np.asarray(prob))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=c,
                 lw=lw, label='{} ({:.2f})'.format(l, roc_auc))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight ='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight ='bold', fontsize=12)
    # plt.title('ROC Curve ')
    plt.legend(loc="lower right")
    if fig_name is not None:
        plt.savefig('figures/{}.pdf'.format(fig_name))
    plt.show()


def plot_hist(taskname, posteriors, priors, save_fig=True):
    n_bins = 10
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 3.5))
    my_blue = (153 / 255, 204 / 255, 1.0, 0.55)
    my_green = (0 / 255, 204 / 255, 102 / 255, 0.6)
    my_edge = (0, 0, 0, 0.55)
    my_green_edge = (0, 153 / 255, 76 / 255, 0.55)
    
    _, bins, _ = ax0.hist(posteriors[0], n_bins, density=True, histtype='bar', facecolor=my_blue, label='Non-hallucinated', edgecolor=my_edge, hatch='//')
    ax0.hist(posteriors[1], bins=bins, density=True, histtype='bar', facecolor=my_green_edge, label='Factual hallucination', edgecolor='darkgreen')
    ax0.set_ylim([0., 6])
    ax0.legend(prop={'size': 11})
    ax0.set_ylabel('CMLM', fontsize=13)
    
    _, bins, _ = ax1.hist(posteriors[0], n_bins, density=True, histtype='bar', facecolor=my_blue, label='Non-hallucinated', edgecolor=my_edge, hatch='//')
    ax1.hist(posteriors[2], bins=bins, density=True, histtype='bar', facecolor='red', label='Non-factual Hallucination', edgecolor='red', alpha=0.50)
#     ax1.set_ylim([0., 6])
    ax1.legend(prop={'size': 11})
    ax1.set_yticklabels([])
    
    _, bins, _ = ax2.hist(priors[0], n_bins, density=True, histtype='bar', facecolor=my_blue, label='Non-hallucinated', edgecolor=my_edge, hatch='//')
    ax2.hist(priors[1], bins=bins, density=True, histtype='bar', facecolor=my_green_edge, label='Factual hallucination', edgecolor='darkgreen')
    ax2.set_ylim([0., 6])
    ax2.set_ylabel('MLM', fontsize=13)
    
    # Make a multiple-histogram of data-sets with different length.
    _, bins, _ = ax3.hist(priors[0], n_bins, density=True, histtype='bar', facecolor=my_blue, label='Non-hallucinated', edgecolor=my_edge, hatch='//')
    ax3.hist(priors[2], bins=bins, density=True, histtype='bar', facecolor='red', label='Non-factual hallucination', edgecolor='red', alpha=0.50)
    ax3.set_yticklabels([])
    
    ax0.tick_params(axis='both', which='major', labelsize=13)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax3.tick_params(axis='both', which='major', labelsize=13)
    
    fig.text(0.5, -0.005, 'Prediction Probability', ha='center', fontsize=13)
    fig.text(0.99, 0.5, 'Density', va='center', rotation='vertical', fontsize=13)
    fig.tight_layout()
    if save_fig:
        plt.savefig("figures/" + taskname +'.pdf', bbox_inches="tight")
    plt.show()