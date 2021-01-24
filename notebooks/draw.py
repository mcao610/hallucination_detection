import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
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
                   [d[1] for d in data], c=colors[i], s=[(d[0] + d[1]) * 100 + 40 for d in data], label=labels[i], alpha=0.7)

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
               edgecolor='black', label=legend_labels[2], alpha=0.8)
    
    ax.legend(loc='lower right')
    ax.set_xlabel(x_label, fontweight='bold', fontsize=11)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=11)
    
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