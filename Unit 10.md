1. **Computation of Various Distance Metrics**
   - **Euclidean Distance**: The most common distance metric, calculated as the square root of the sum of the squared differences between the individual dimensions of two points. \( \text{Euclidean}(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} \).
   - **Manhattan Distance**: The sum of the absolute differences of their coordinates. \( \text{Manhattan}(x, y) = \sum_{i=1}^{n} |x_i - y_i| \).
   - **Minkowski Distance**: A generalization of Euclidean and Manhattan distances. \( \text{Minkowski}(x, y) = (\sum_{i=1}^{n} |x_i - y_i|^p)^{1/p} \), where \( p \) is the order of the norm.
   - **Cosine Similarity**: Measures the cosine of the angle between two vectors, used as a similarity metric. \( \text{Cosine}(x, y) = \frac{x \cdot y}{\|x\| \|y\|} \), where \( \cdot \) denotes the dot product and \( \|x\| \) is the norm of vector \( x \).

2. **Concept of Dendrogram**
   - A dendrogram is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering. Each branch represents a cluster and its length represents the degree of similarity (or dissimilarity) between clusters. It provides a visual summary of the clustering process, helping to determine the number of clusters by showing the points at which clusters are merged.

3. **Agglomerative vs. Divisive Hierarchical Clustering**
   - **Agglomerative (Bottom-Up)**: Starts with each data point as a single cluster and then merges the closest pairs of clusters until all points have been merged into a single cluster.
   - **Divisive (Top-Down)**: Starts with all data points in a single cluster and recursively splits the most heterogeneous cluster until each data point forms a cluster.

4. **Applications of Clustering Algorithms**
   - **Market Segmentation**: Clustering can identify distinct groups within a customer base, helping businesses tailor marketing strategies to specific segments.
   - **Document Clustering**: In information retrieval or text mining, clustering is used to group similar documents together, which helps in organizing, summarizing, and searching information.

5. **Different Linkage Methods in Clustering**
   - **Single Linkage**: The distance between two clusters is defined as the shortest distance between any two points in the clusters. Example: Useful in identifying elongated or stringy clusters.
   - **Complete Linkage**: The distance between two clusters is defined as the longest distance between any two points in the clusters. Example: Tends to find compact clusters of approximately equal diameters.
   - **Average Linkage**: The distance between two clusters is the average distance between each point in one cluster to every point in the other cluster.
   - **Ward’s Method**: Minimizes the total within-cluster variance. At each step, the pair of clusters with the minimum between-cluster distance are merged.