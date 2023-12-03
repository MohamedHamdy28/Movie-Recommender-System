import torch
from sklearn.metrics import mean_squared_error
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, test_loader, k=5):
    """
    Evaluates the model on the test dataset.

    Args:
    model (torch.nn.Module): The trained model.
    test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
    k (int): The number of top recommendations to consider for precision.

    Returns:
    float: RMSE value.
    float: Precision at top-k value.
    """
    model.eval()
    all_preds = []
    all_actuals = []
    top_k_precision = []

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            # Predict ratings
            preds = model(user_ids, item_ids).cpu().numpy()
            actuals = ratings.cpu().numpy()

            all_preds.extend(preds)
            all_actuals.extend(actuals)

            # Calculate precision at top k
            for uid in torch.unique(user_ids):
                user_mask = user_ids == uid
                user_preds = preds[user_mask]
                user_actuals = actuals[user_mask]

                top_items = user_preds.argsort()[-k:][::-1]
                top_items_actuals = user_actuals[top_items]

                relevant_items = sum(top_items_actuals >= 4)  # Assuming 4 and above are relevant ratings
                precision = relevant_items / k
                top_k_precision.append(precision)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(all_actuals, all_preds))

    # Calculate average precision at top k
    average_precision_at_k = sum(top_k_precision) / len(top_k_precision)

    return rmse, average_precision_at_k