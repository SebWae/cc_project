import onnx
import torch
import torchvision

# my_model = torchvision.models.resnet18(weights=None)
# # model.eval()
# dummy_input = torch.randn(1, 3, 224, 224)

# # exporting the model to ONNX
# onnx_model = torch.onnx.dynamo_export(
#     my_model,
#     model_args=(dummy_input,),
#     export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
# )
# onnx_model.save("resnet18.onnx")

class MyModel(torch.nn.Module):
    def init(self) -> None:
        super().init()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x, bias=None):
        out = self.linear(x)
        out = out + bias
        return out

model = MyModel()

args = (torch.randn(2, 2, 2),)
   

export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(
    model,
    *args,
    # **kwargs,
    export_options=export_options)
onnx_program.save("my_dynamic_model.onnx")

# loading the model from ONNX
model = onnx.load("my_simple_model.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))