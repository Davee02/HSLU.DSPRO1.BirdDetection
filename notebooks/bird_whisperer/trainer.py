import json
import os
from sklearn.metrics import classification_report, accuracy_score
import torch
from tqdm import tqdm


def train(device, model, train_dataloader, test_dataloader, criterion, optimizer, unique_labels, start_epoch, n_epochs, save_model_path, json_log_path):
  best_macro_avg_f1 = 0
  best_epoch = 0
  train_losses = []
  test_epochs = []
  test_metrics = []

  os.makedirs(save_model_path, exist_ok=True)

  for epoch in range(start_epoch, n_epochs):
    # Train the model
    train_classification_report, avg_epoch_loss = run_epoch(device, model, train_dataloader, criterion, optimizer, unique_labels, epoch)
    macro_avg_f1_train = train_classification_report["macro avg"]["f1-score"]

    train_losses.append(avg_epoch_loss)
    print(f"Train Epoch: {epoch}, Macro Avg F1: {macro_avg_f1_train:0.4f}, Avg Loss: {avg_epoch_loss:0.4f}")

    # Save the model
    save_model(model, optimizer, epoch, avg_epoch_loss, save_model_path)

    # Test the model
    print(f"\nTesting the model after Epoch {epoch}...")
    test_classification_report, avg_test_loss = test_model(device, model, test_dataloader, criterion, unique_labels)
    macro_avg_f1_test = test_classification_report["macro avg"]["f1-score"]
    print_test_scores(test_classification_report, avg_test_loss)

    test_metrics.append([test_classification_report['macro avg']['f1-score'], test_classification_report['macro avg']['precision'], test_classification_report['macro avg']['recall'], test_classification_report['accuracy'], avg_test_loss])
    test_epochs.append(epoch)

    # Save the best model based on macro_avg_f1 score
    if macro_avg_f1_test > best_macro_avg_f1:
        best_macro_avg_f1 = macro_avg_f1_test
        best_epoch = epoch
        print(f"\n{'Best Macro Avg F1-Score':<20} = {best_macro_avg_f1:0.4f}")
        print(f"Saving the best model at '{save_model_path}' ... ")
        save_model(model, optimizer, epoch, avg_epoch_loss, save_model_path, "best_model.pt")

    results = {
      "train_epoch": list(range(start_epoch, epoch + 1)),
      "train_loss": train_losses,
      "test_epoch": test_epochs,
      "epoch_test_metrics": test_metrics
    }

    with open(json_log_path, 'w') as f:
      json.dump(results, f)

  print(f"\nBest Macro Avg F1-Score = {best_macro_avg_f1:0.4f} was achieved at Epoch = {best_epoch}\n")

def run_epoch(device, model, train_dataloader, criterion, optimizer, unique_labels, epoch):
  print(f"\n\n============================= Epoch: {epoch} =============================")

  model.train()

  actual_labels = []
  predicted_labels = []
  loss_vals = []

  for batch_idx, batch in enumerate(tqdm(train_dataloader)):
    if batch_idx > 5:
      break
    log_mels, labels, _ = batch
    log_mels, labels = log_mels.float().to(device), labels.to(device)

    optimizer.zero_grad() # zero the gradiants of the parameters
    logits = model(log_mels) # forward pass through the model
    loss = criterion(logits, labels) # compute loss
    loss.backward() # compute gradients of the parameters
    optimizer.step() # update the weights with gradients

    _, preds = torch.max(logits, 1)
    predicted_labels.extend(preds.cpu().detach().numpy())
    actual_labels.extend(labels.cpu().detach().numpy())
    loss_vals.append(loss.item())

    if batch_idx % 100 == 0:
      print(f"Batch Index = {batch_idx:03}, Loss = {loss.item():0.4f}")

  avg_epoch_loss = sum(loss_vals) / len(loss_vals)
  class_report = classification_report(actual_labels, predicted_labels, labels=unique_labels, zero_division=0, output_dict=True)
  if "accuracy" not in class_report:
    class_report["accuracy"] = accuracy_score(actual_labels, predicted_labels)
  
  return class_report, avg_epoch_loss

def test_model(device, model, test_dataloader, criterion, unique_labels):
    with torch.no_grad():
      model.eval()

      actual_labels = []
      predicted_labels = []
      loss_vals = []

      for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        if batch_idx > 5:
          break

        log_mels, labels, _ = batch
        log_mels, labels = log_mels.float().to(device), labels.to(device)

        logits = model(log_mels) # forward pass through the model
        loss = criterion(logits, labels) # compute loss

        _, preds = torch.max(logits, 1)
        predicted_labels.extend(preds.cpu().detach().numpy())
        actual_labels.extend(labels.cpu().detach().numpy())
        loss_vals.append(loss.item())

      avg_test_loss = sum(loss_vals) / len(loss_vals)
      class_report = classification_report(actual_labels, predicted_labels, labels=unique_labels, zero_division=0, output_dict=True)
      if "accuracy" not in class_report:
        class_report["accuracy"] = accuracy_score(actual_labels, predicted_labels)
      
      return class_report, avg_test_loss

def save_model(model, optimizer, epoch, epoch_avg_loss, save_model_path, model_name=None):
  model_name = model_name if model_name is not None else f"checkpoint_epoch_{epoch}_loss_{epoch_avg_loss:0.4f}.pt"
  save_path = os.path.join(save_model_path, model_name)

  torch.save({
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "epoch_avg_loss": epoch_avg_loss
  }, save_path)

def print_test_scores(classification_report_test, avg_test_loss):
  print("\n\n=============== Test Metrics ===============")
  print(f"Average Loss = {avg_test_loss:0.4f}")

  summary = {
        'accuracy': classification_report_test['accuracy'],
        'macro avg': {
            'precision': classification_report_test['macro avg']['precision'],
            'recall': classification_report_test['macro avg']['recall'],
            'f1-score': classification_report_test['macro avg']['f1-score']
        },
        'weighted avg': {
            'precision': classification_report_test['weighted avg']['precision'],
            'recall': classification_report_test['weighted avg']['recall'],
            'f1-score': classification_report_test['weighted avg']['f1-score']
        }
    }
  
  print("\nAccuracy: {:.4f}".format(summary['accuracy']))
  
  print("\nMacro Average:")
  print("  Precision: {:.4f}".format(summary['macro avg']['precision']))
  print("  Recall: {:.4f}".format(summary['macro avg']['recall']))
  print("  F1-Score: {:.4f}".format(summary['macro avg']['f1-score']))

  print("\nWeighted Average:")
  print("  Precision: {:.4f}".format(summary['weighted avg']['precision']))
  print("  Recall: {:.4f}".format(summary['weighted avg']['recall']))
  print("  F1-Score: {:.4f}".format(summary['weighted avg']['f1-score']))
  print("\n==============================================\n\n")
