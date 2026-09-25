import numpy as np
import matplotlib.pyplot as plt


datasets = {
    "BlogCatalog": "singular_values_blogcatalog.npy",
    "PPI": "singular_values_ppi.npy",
    "Wikipedia": "singular_values_wikipedia.npy",
}

for name, filename in datasets.items():
    singular_values = np.load(filename)

    plt.plot(
        range(1, len(singular_values) + 1),
        singular_values,
        label=name,
    )

plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")
plt.title("NetMF Singular Value Decay")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "singular_value_decay.png",
    dpi=300,
)

plt.show()