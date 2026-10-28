import torch


class Trainer:
    def __init__(self, device, loss_function, optimizer, max_epochs):
        self.device = device
        self.loss_func = loss_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.history = []

    def fit(self, model, train_loader, val_loader, callbacks):
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.on_train_begin({})

        model.to(self.device)
        for epoch in range(self.max_epochs):
            for callback in self.callbacks:
                callback.on_train_loop_begin({})

            train_logs = self.train_epoch(model, train_loader)
            val_logs = self.validate_epoch(model, val_loader)
            self.history.append({"epoch": epoch + 1, **train_logs, **val_logs})

            metrics = {**train_logs, **val_logs}
            logs = {
                "epoch": epoch,
                "total_epochs": self.max_epochs,
                "metrics": metrics,
                "model": model,
                "optimizer": self.optimizer,
            }

            should_stop = False
            for callback in self.callbacks:
                res = callback.on_train_loop_end(logs)

                if isinstance(res, dict) and "should_stop" in res.keys():
                    should_stop = res["should_stop"]

            if should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end({})

        return self.history

    def evaluate(self, model, dataloader, callbacks=[]):
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.on_evaluate_begin({})

        logs = self.test_epoch(model, dataloader)
        metrics = {
            "test_loss": logs["test_loss"],
            "test_acc": logs["test_acc"],
        }

        logs = {"metrics": metrics}
        for callback in self.callbacks:
            callback.on_evaluate_end(logs)
        return logs

    def save_model(self, model, save_path):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def train_epoch(self, model, dataloader):
        logs = {
            "total_batchs": len(dataloader),
            "description": "Training...",
        }
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            logs = {"total_batchs": len(dataloader)}
            for callback in self.callbacks:
                callback.on_batch_begin(logs)

            data, labels = data.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = model(data.float())
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            logs = {
                "loss": loss.item(),
                "acc": (correct / total) if total else 0.0,
            }
            for callback in self.callbacks:
                callback.on_batch_end(logs)

        logs = {
            "train_loss": total_loss / len(dataloader),
            "train_acc": correct / total,
        }
        for callback in self.callbacks:
            callback.on_epoch_end(logs)
        return logs

    def validate_epoch(self, model, dataloader):
        logs = {
            "total_batchs": len(dataloader),
            "description": "Validating...",
        }
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)
        model.eval()

        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                logs = {"total_batchs": len(dataloader)}
                for callback in self.callbacks:
                    callback.on_batch_begin(logs)

                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data.float())
                loss = self.loss_func(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                logs = {
                    "loss": loss.item(),
                    "acc": (correct / total) if total else 0.0,
                }
                for callback in self.callbacks:
                    callback.on_batch_end(logs)

        logs = {
            "val_loss": total_loss / len(dataloader),
            "val_acc": correct / total,
        }
        for callback in self.callbacks:
            callback.on_epoch_end(logs)
        return logs

    def test_epoch(self, model, test_loader):
        logs = {
            "total_batchs": len(test_loader),
            "description": "Evaluating...",
        }
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                for callback in self.callbacks:
                    callback.on_batch_begin({})

                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data.float())
                loss = self.loss_func(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                logs = {
                    "loss": loss.item(),
                    "acc": (correct / total) if total else 0.0,
                }
                for callback in self.callbacks:
                    callback.on_batch_end(logs)

        logs = {
            "test_loss": total_loss / len(test_loader),
            "test_acc": correct / total if total else 0.0,
        }
        for callback in self.callbacks:
            callback.on_epoch_end(logs)
        return logs
