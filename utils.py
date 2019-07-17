import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)


# gan validation function
def validation(val_out, val_block_size, image_fn):
    preprocesed = ((val_out + 1.0) * 127.5).astype(np.uint8)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    # save image
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    image = Image.fromarray(final_image)
    image.save(image_fn)
    return


# save losses
def save_losses(losses, labels, elapsed_time, fn):
    fig, ax = plt.subplots()
    losses = np.array(losses)

    for col in range(losses.shape[1]):
        plt.plot(losses.T[col], label=labels[col], alpha=0.5)

    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))
    elapsed_time_fn = 'elapsed: {:.3f}s'.format(elapsed_time)
    plt.text(0.2, 0.9, elapsed_time_fn, ha='center', va='center', transform=ax.transAxes)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(fn)
    plt.close(fig)
    return
