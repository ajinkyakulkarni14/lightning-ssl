import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class KNNEvaluator:
    """KNNEvaluator class
    """
    def __init__(self) -> None:
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        
    def reset(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def _update_train(self, embeds: torch.Tensor, targets: torch.Tensor):
        """updates train data structures

        Args:
            embeds (torch.Tensor): model embeds
            targets (torch.Tensor): batch targets
        """
        
        for embed, target in zip(embeds, targets):
            self.x_train.append(embed.detach().cpu().numpy())
            self.y_train.append(target.detach().cpu().numpy())
            
    def _update_val(self, embeds: torch.Tensor, targets: torch.Tensor):
        """updates val data structures

        Args:
            embeds (torch.Tensor): model embeds
            targets (torch.Tensor): batch targets
        """
        for embed, target in zip(embeds, targets):
            self.x_test.append(embed.detach().cpu().numpy())
            self.y_test.append(target.detach().cpu().numpy())
  
    def update(self, split: str, embeds: torch.Tensor, targets: torch.Tensor):
        """update train/val data structures

        Args:
            split (str): train/val
            embeds (torch.Tensor): model embeds
            targets (torch.Tensor): batch targets
        """
        if split == "train": self._update_train(embeds, targets)
        if split == "val": self._update_val(embeds, targets)
        
    def compute(self) -> float:
        """comptues accuracy by first training a KNeighborsClassifier on train+val data structures

        Returns:
            float: accuracy
        """
        if len(self.x_train) == 0:
            return 0.0
        
        estimator = KNeighborsClassifier()
        estimator.fit(self.x_train, self.y_train)
        y_preds = estimator.predict(self.x_test)
        acc = accuracy_score(self.y_test, y_preds)
        return acc
            
        
        
        
        
        