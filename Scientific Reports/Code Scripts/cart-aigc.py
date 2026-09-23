import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor, export_text
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# ==================== Stage 1: CGAN Digital Twin Module ====================

class Generator(nn.Module):
    """CGAN Generator"""
    def __init__(self, z_dim: int = 128, c_dim: int = 85, x_dim: int = 256):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, x_dim)  # No activation in output layer
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Generate synthetic preference vectors"""
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    """CGAN Discriminator"""
    def __init__(self, x_dim: int = 256, c_dim: int = 85):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Discriminate the probability that input samples come from the real distribution"""
        return self.net(torch.cat([x, c], dim=1))


class UserDigitalTwin:
    """User Cognitive Digital Twin (CGAN Wrapper)"""

    def __init__(self, z_dim: int = 128, c_dim: int = 85, x_dim: int = 256,
                 lr_g: float = 2e-4, lr_d: float = 1e-4, batch_size: int = 256):
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim

        self.generator = Generator(z_dim, c_dim, x_dim)
        self.discriminator = Discriminator(x_dim, c_dim)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self, D_seed: np.ndarray, style_labels: np.ndarray,
              epochs: int = 500, patience: int = 20) -> Dict[str, List[float]]:
        """
        Train CGAN Digital Twin

        Args:
            D_seed: Real seed dataset (n_samples, x_dim)
            style_labels: Style labels (n_samples, c_dim)
            epochs: Maximum training epochs
            patience: Early stopping patience

        Returns:
            Training history records
        """
        n_samples = D_seed.shape[0]
        D_seed_tensor = torch.FloatTensor(D_seed).to(self.device)
        style_tensor = torch.FloatTensor(style_labels).to(self.device)

        real_label = 0.9   # Label smoothing
        fake_label = 0.1

        history = {'g_loss': [], 'd_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # ---------- Train Discriminator ----------
            self.discriminator.zero_grad()

            # Real samples
            idx = np.random.choice(n_samples, self.batch_size)
            real_x = D_seed_tensor[idx]
            real_c = style_tensor[idx]
            real_labels = torch.full((self.batch_size, 1), real_label, device=self.device)

            real_output = self.discriminator(real_x, real_c)
            d_loss_real = self.criterion(real_output, real_labels)

            # Generated samples
            z = torch.randn(self.batch_size, self.z_dim, device=self.device)
            fake_c = style_tensor[np.random.choice(n_samples, self.batch_size)]
            fake_x = self.generator(z, fake_c).detach()
            fake_labels = torch.full((self.batch_size, 1), fake_label, device=self.device)

            fake_output = self.discriminator(fake_x, fake_c)
            d_loss_fake = self.criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_d.step()

            # ---------- Train Generator ----------
            self.generator.zero_grad()

            z = torch.randn(self.batch_size, self.z_dim, device=self.device)
            fake_c = style_tensor[np.random.choice(n_samples, self.batch_size)]
            fake_x = self.generator(z, fake_c)
            output = self.discriminator(fake_x, fake_c)

            # Generator objective: make discriminator believe generated samples are real
            g_loss = self.criterion(output, torch.full((self.batch_size, 1), real_label, device=self.device))
            g_loss.backward()
            self.optimizer_g.step()

            # ---------- Logging & Early Stopping ----------
            history['g_loss'].append(g_loss.item())
            history['d_loss'].append(d_loss.item())

            # Simple validation (simplified here)
            val_loss = g_loss.item() + d_loss.item()
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return history

    def synthesize(self, n_users: int, style_labels: np.ndarray) -> np.ndarray:
        """Synthesize large-scale virtual user preference data"""
        self.generator.eval()
        synthetic_data = []

        with torch.no_grad():
            for i in range(0, n_users, self.batch_size):
                batch_size = min(self.batch_size, n_users - i)
                z = torch.randn(batch_size, self.z_dim, device=self.device)

                # Randomly sample conditions from style labels
                idx = np.random.choice(len(style_labels), batch_size)
                c = torch.FloatTensor(style_labels[idx]).to(self.device)

                fake_x = self.generator(z, c)
                synthetic_data.append(fake_x.cpu().numpy())

        return np.vstack(synthetic_data)


# ==================== Stage 2: CART Rule Extraction Module ====================

class CARTModel:
    """CART Decision Tree Wrapper"""

    def __init__(self, max_depth: int = 12, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.rules = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train CART Decision Tree

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Preference scores (n_samples,)
        """
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion='squared_error',  # Regression tree uses MSE
            ccp_alpha=0.0  # Cost-complexity pruning can be applied later
        )
        self.model.fit(X, y)
        self._extract_rules()

    def _extract_rules(self) -> None:
        """Extract if-then rule set from decision tree"""
        tree_text = export_text(self.model, feature_names=[f'f{i}' for i in range(self.model.n_features_in_)])
        self.rules = tree_text

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict preference scores"""
        return self.model.predict(X)

    def get_leaf_rules(self) -> List[Tuple[str, float]]:
        """
        Get leaf node rule list

        Returns:
            List of (rule_path, predicted_value)
        """
        n_nodes = self.model.tree_.node_count
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        value = self.model.tree_.value

        leaf_rules = []

        def traverse(node_id: int, path: str):
            if children_left[node_id] == -1 and children_right[node_id] == -1:
                # Leaf node
                leaf_rules.append((path, value[node_id][0][0]))
                return

            # Internal node
            feat_name = f'f{feature[node_id]}'
            thresh = threshold[node_id]

            # Left branch
            traverse(children_left[node_id], f"{path} AND {feat_name} <= {thresh:.4f}" if path else f"{feat_name} <= {thresh:.4f}")
            # Right branch
            traverse(children_right[node_id], f"{path} AND {feat_name} > {thresh:.4f}" if path else f"{feat_name} > {thresh:.4f}")

        traverse(0, "")
        return leaf_rules


# ==================== Stage 3: Closed-Loop Iterative Optimization Module ====================

class DiffusionModelWrapper:
    """Diffusion Model Wrapper (Simulated)"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.theta = None  # Model parameters

    def sample(self, rules: List[Tuple[str, float]],
               num_samples: int = 10) -> np.ndarray:
        """
        Sample content from diffusion model with CART rules as prior constraints

        Args:
            rules: CART rule set (if-then rules, predicted values)
            num_samples: Number of samples to generate

        Returns:
            Generated artwork feature vectors
        """
        # Simulate diffusion model generation process
        # In practice, this should call Stable Diffusion or similar models
        np.random.seed(42)

        # Adjust generation direction according to rule constraints
        feature_dim = 256
        samples = np.random.randn(num_samples, feature_dim) * 0.5

        # Simulate the influence of rule constraints on generation
        for rule, pred_val in rules[:3]:  # Take first 3 rules as constraints
            # Rule constraints manifest as shifting toward predicted value
            strength = 0.15  # lambda parameter
            offset = pred_val / 100.0
            samples += offset * strength * np.ones_like(samples)

        return samples


class UserAgent:
    """Virtual User Agent"""

    def __init__(self, n_agents: int = 10000, feature_dim: int = 256):
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.preference_vectors = None
        self.alpha = np.random.uniform(0.5, 1.5, n_agents)
        self.beta = np.random.uniform(-0.5, 0.5, n_agents)

    def load_preferences(self, pref_vectors: np.ndarray) -> None:
        """Load user preference vectors"""
        self.preference_vectors = pref_vectors

    def evaluate(self, X_gen: np.ndarray) -> np.ndarray:
        """
        Score generated content

        Args:
            X_gen: Generated content feature vectors (n_samples, feature_dim)

        Returns:
            Score array (n_samples,)
        """
        if self.preference_vectors is None:
            raise ValueError("Please load user preference vectors first")

        scores = []
        for x in X_gen:
            # Compute weighted similarity average with all user agents
            sim = np.dot(self.preference_vectors, x)
            avg_sim = np.mean(sim)
            score = 1.0 / (1.0 + np.exp(-(0.8 * avg_sim + 0.1)))  # sigmoid mapping to [0,1]
            scores.append(score)

        return np.array(scores)


# ==================== Main Algorithm: CART-AIGC Collaborative Generation ====================

def cart_aigc_algorithm(
    D_seed: np.ndarray,           # Real seed dataset
    S: np.ndarray,                # Style label set
    N: int = 50,                  # Maximum iteration rounds
    epsilon: float = 0.01,        # Convergence threshold
    n_users: int = 10000,         # Number of synthetic users
    n_labels: int = 500,          # Number of style labels per user
    max_depth: int = 12,          # CART max depth
    min_samples_leaf: int = 20    # CART min leaf samples
) -> Tuple[CARTModel, Dict[str, Any]]:
    """
    CART-AIGC Collaborative Generation Algorithm - Main Function

    Args:
        D_seed: Real seed dataset (n_seed, x_dim)
        S: Style label set (n_labels, c_dim)
        N: Maximum closed-loop iteration rounds
        epsilon: Convergence threshold
        n_users: Number of synthetic virtual users
        n_labels: Number of style labels per user
        max_depth: CART max depth
        min_samples_leaf: CART min leaf samples

    Returns:
        R: Optimal CART model (including rule set)
        theta: Optimal generation model parameters
    """

    # ==================== Stage 1: Data Synthesis (CGAN Digital Twin) ====================
    print("Stage 1: CGAN Digital Twin Data Synthesis...")

    # Initialize CGAN
    twin = UserDigitalTwin()

    # Train CGAN
    history = twin.train(D_seed, S, epochs=500)

    # Synthesize large-scale user data
    # Generate n_users virtual users, each scoring n_labels style labels
    D_syn = twin.synthesize(n_users, S)

    # Validate distribution alignment (KL divergence < epsilon_KL and MMD < epsilon_MMD)
    print("Validating distribution alignment between synthetic and real data...")
    # Simplified validation: compute KL divergence and MMD
    def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        # Discretize then compute KL divergence
        p_hist, bins = np.histogram(p.flatten(), bins=50, density=True)
        q_hist, _ = np.histogram(q.flatten(), bins=bins, density=True)
        p_hist += 1e-10
        q_hist += 1e-10
        return np.sum(p_hist * np.log(p_hist / q_hist))

    def compute_mmd(p: np.ndarray, q: np.ndarray, sigma: float = 1.0) -> float:
        # Simplified MMD computation (RBF kernel)
        def rbf_kernel(x, y, sigma=1.0):
            return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

        # Random subset sampling for computation
        idx_p = np.random.choice(len(p), min(100, len(p)), replace=False)
        idx_q = np.random.choice(len(q), min(100, len(q)), replace=False)
        p_sample = p[idx_p]
        q_sample = q[idx_q]

        k_pp = np.mean([rbf_kernel(p_sample[i], p_sample[j]) for i in range(len(p_sample)) for j in range(len(p_sample))])
        k_qq = np.mean([rbf_kernel(q_sample[i], q_sample[j]) for i in range(len(q_sample)) for j in range(len(q_sample))])
        k_pq = np.mean([rbf_kernel(p_sample[i], q_sample[j]) for i in range(len(p_sample)) for j in range(len(q_sample))])

        return k_pp + k_qq - 2 * k_pq

    kl_div = compute_kl_divergence(D_seed, D_syn)
    mmd_val = compute_mmd(D_seed, D_syn)

    print(f"KL Divergence: {kl_div:.4f}, MMD: {mmd_val:.4f}")

    # If distribution alignment validation fails, perform additional training
    if kl_div >= 0.05:
        print(f"Warning: KL divergence ({kl_div:.4f}) >= 0.05, continuing CGAN training...")
        # In practice, you could increase training epochs or tune parameters here

    # ==================== Stage 2: CART Rule Extraction ====================
    print("Stage 2: CART Rule Extraction...")

    # Build training data: X as synthetic user data, y as user preference scores
    # Simplified here: take the first half of D_syn as features, second half as labels
    # In practice, separate features and labels from D_syn
    X_syn = D_syn[:, :128]  # Feature dimension
    y_syn = D_syn[:, 128]   # Preference score dimension

    # Train CART
    cart = CARTModel(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    cart.train(X_syn, y_syn)

    # Extract rule set
    leaf_rules = cart.get_leaf_rules()
    print(f"Extracted {len(leaf_rules)} decision rules")
    print("Sample rules:", leaf_rules[:3])

    # ==================== Stage 3: Closed-Loop Iterative Optimization ====================
    print("Stage 3: Closed-Loop Iterative Optimization...")

    # Initialize diffusion model (using simulated wrapper here)
    diffusion = DiffusionModelWrapper()

    # Initialize user agents (for simulated scoring)
    agent = UserAgent(n_agents=n_users)
    agent.load_preferences(D_syn)  # Use synthetic data as user preference baseline

    R = cart  # Current CART model
    theta = None  # Generation model parameters

    # Training logs
    training_log = {
        'loss_history': [],
        'score_history': [],
        'rules_history': []
    }

    for t in range(1, N + 1):
        print(f"Iteration round {t}/{N}")

        # Use R as prior constraint, sample content from diffusion model
        X_gen = diffusion.sample(leaf_rules, num_samples=10)

        # User agent scoring
        scores = agent.evaluate(X_gen)

        # Compute deviation (MSE)
        y_pred = R.predict(X_gen)
        loss = np.mean((scores - y_pred) ** 2)
        training_log['loss_history'].append(loss)
        training_log['score_history'].append(np.mean(scores))

        print(f"  Loss: {loss:.6f}, Average Score: {np.mean(scores):.4f}")

        # Convergence check
        if loss < epsilon:
            print(f"Loss {loss:.6f} < threshold {epsilon}, converged!")
            break

        # Feedback optimization: update CART rules
        # Use generated results and scores as new data points, incrementally update CART model
        # Simplified here: add new data to training set and retrain
        X_combined = np.vstack([X_syn, X_gen])
        y_combined = np.concatenate([y_syn, scores])
        R.train(X_combined, y_combined)
        leaf_rules = R.get_leaf_rules()
        training_log['rules_history'].append(leaf_rules)

    print(f"\nCollaborative generation completed! Total iterations: {t}")

    return R, theta


# ==================== Example Run ====================

if __name__ == "__main__":
    # Generate simulated dataset
    np.random.seed(42)

    # Seed dataset: 50 real user samples
    n_seed = 50
    x_dim = 256
    c_dim = 85

    D_seed = np.random.randn(n_seed, x_dim) * 0.5 + 0.5
    S = np.random.randn(500, c_dim) * 0.3

    # Run algorithm
    R_optimal, theta_optimal = cart_aigc_algorithm(
        D_seed=D_seed,
        S=S,
        N=50,
        epsilon=0.01,
        n_users=10000,
        n_labels=500
    )

    print("\n" + "=" * 60)
    print("Algorithm execution completed!")
    print("=" * 60)
    print(f"Optimal rule set sample:\n{R_optimal.rules[:500]}")  # Print first 500 chars