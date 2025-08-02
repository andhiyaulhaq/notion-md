# Modifying YOLO Architecture

modify CBAM (class already defined by yolo)

copy ultralytics/cfg/models/v8/yolov8.yaml

paste to root dir and rename it as yolov8_customized.yaml

- edit yolov8_customized.yaml
    
    ```yaml
    # Ultralytics 🚀 AGPL-3.0 License
    # Modified YOLOv8 with CBAM attention
    
    # Parameters
    nc: 80
    scales:
      n: [0.33, 0.25, 1024]
      s: [0.33, 0.50, 1024]
      m: [0.67, 0.75, 768]
      l: [1.00, 1.00, 512]
      x: [1.00, 1.25, 512]
    
    # Backbone with CBAM
    backbone:
      - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
      - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
      - [-1, 3, C2f, [128, True]] # 2
      - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
      - [-1, 6, C2f, [256, True]] # 4
      - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
      - [-1, 6, C2f, [512, True]] # 6
      - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
      - [-1, 3, C2f, [1024, True]] # 8
      - [-1, 1, CBAM, [1024]] # 9 (NEW CBAM layer)
      - [-1, 1, SPPF, [1024, 5]] # 10
    
    # Head (adjusted indices)
    head:
      - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 11
      - [[-1, 6], 1, Concat, [1]] # 12 (cat backbone P4)
      - [-1, 3, C2f, [512]] # 13
    
      - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 14
      - [[-1, 4], 1, Concat, [1]] # 15 (cat backbone P3)
      - [-1, 3, C2f, [256]] # 16 (P3/8-small)
    
      - [-1, 1, Conv, [256, 3, 2]] # 17
      - [[-1, 13], 1, Concat, [1]] # 18 (cat head P4)
      - [-1, 3, C2f, [512]] # 19 (P4/16-medium)
    
      - [-1, 1, Conv, [512, 3, 2]] # 20
      - [[-1, 10], 1, Concat, [1]] # 21 (cat head P5 - UPDATED index)
      - [-1, 3, C2f, [1024]] # 22 (P5/32-large)
    
      - [[16, 19, 22], 1, Detect, [nc]] # 23 (UPDATED indices)
    
    ```
    

duplicate it to be yolov8s_customized.yaml and yolov8n_customized.yaml

- edit ultralytics/nn/modules/conv.py (modify CBAM)
    
    ```python
    class CBAM(nn.Module):
        """Convolutional Block Attention Module"""
    
        def __init__(self, c1, reduction=16):
            super().__init__()
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1 // reduction, 1),
                nn.ReLU(),
                nn.Conv2d(c1 // reduction, c1, 1),
                nn.Sigmoid(),
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
            )
    
        def forward(self, x):
            # Channel attention
            ca = self.channel_attention(x)
            x_out = x * ca
    
            # Spatial attention
            max_pool = torch.max(x_out, dim=1, keepdim=True)[0]
            avg_pool = torch.mean(x_out, dim=1, keepdim=True)
            sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
            return x_out * sa
    ```
    
- edit ultralytics/nn/modules/tasks.py
    
    ```python
    from ultralytics.nn.modules import (
    	...,
    	CBAM,
    	...
    )
    
    base_modules = frozenset({
    	...,
    	Conv,
    	CBAM,
    	...
    })
    
    if m in base_modules:
    	c1, c2 = ...
    ```
    
- create nyoba.ipynb
    
    ```python
    from ultralytics import YOLO
    
    # Load custom config
    model_s = YOLO('yolov8s_customized.yaml')  # Use your modified YAML
    model_n = YOLO('yolov8n_customized.yaml')  # Use your modified YAML
    
    print(f"Using scale: {model_s.model.yaml['scale']}")  # Should be 's'
    print(f"Using scale: {model_n.model.yaml['scale']}")  # Should be 'n'
    
    # Start training
    model_s.train(data='ultralytics/cfg/datasets/coco128.yaml', epochs=1, imgsz=640)
    ```
    

create custom block

- edit ultralytics/nn/modules/conv.py
    
    ```python
    class EConv(nn.Module):
        """Enhanced Convolution: Conv2d + BatchNorm + SiLU + SE Attention"""
    
        def __init__(self, c1, c2, k=3, s=1, p=None, r=16):
            # Ensure all parameters are integers
            c1 = int(c1)
            c2 = int(c2)
            k = int(k)
            s = int(s)
    
            # Auto-pad if not specified
            if p is None:
                p = k // 2
            else:
                p = int(p)
    
            super().__init__()
    
            # 1. Main convolution path
            self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU()
    
            # 2. SE Attention
            reduced_channels = max(1, int(c2 // r))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, reduced_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(reduced_channels, c2, kernel_size=1),
                nn.Sigmoid(),
            )
    
            print(f"Created EConv: in={c1}, out={c2}, k={k}, s={s}")
    
        def forward(self, x):
            x = self.act(self.bn(self.conv(x)))
            se = self.se(x)
            return x * se
    ```
    
- edit ultralytics/nn/modules/tasks.py
    
    ```python
    from ultralytics.nn.modules import (
    	...
    	EConv
    )
    
    base_modules = frozenset({
    	...,
    	Conv,
    	EConv,
    	...
    })
    
    if m in base_modules:
    	if m is EConv:
    		c1, c2 = ch[f], args[0]
    		args = [
    			int(x) if isinstance(x, (float, str)) and x.isdigit()
    			else x
    			for x in args
    		]
    	else:
    		c1, c2 = ch[f], args[0]
    ```
    
- edit ultralytics/nn/modules/__init__.py
    
    ```python
    from .conv import (
    	...,
    	EConv
    )
    
    __all__ = (
    	...,
    	"Econv"
    )
    ```
    
- create yolov8n_econv_classify.yaml
    
    ```python
    # yolov8n_econv.yaml
    task: classify
    nc: 10
    imgsz: 64 # Add this to define input size
    
    scales:
      n: [0.33, 0.25, 1024]
      s: [0.33, 0.50, 1024]
      m: [0.67, 0.75, 768]
      l: [1.00, 1.00, 512]
      x: [1.00, 1.25, 512]
    
    # Model configuration
    backbone:
      - [-1, 1, EConv, [8, 8, 8]] # in=3, out=8, k=8, s=8
      - [-1, 1, Conv, [8, 3, 2]] # in=8, out=8, k=3, s=2
    
    head:
      - [-1, 1, nn.AdaptiveAvgPool2d, [1]]
      - [-1, 1, nn.Flatten, []]
      - [-1, 1, nn.Linear, [8, 10]] # 8 input features, 10 output
    ```
    
- create yolov8n_econv_detect.yaml
    
    ```python
    # yolov8n_econv_detect.yaml
    task: detect
    nc: 80
    imgsz: 640
    
    scales:
      n: [0.33, 0.25, 1024]
      s: [0.33, 0.50, 1024]
      m: [0.67, 0.75, 768]
      l: [1.00, 1.00, 512]
      x: [1.00, 1.25, 512]
    
    backbone:
      # EConv arguments adjusted to: [kernel_size, stride, output_channels]
      # Original: [16, 3, 2] (ch_out, k, s)
      # Corrected: [3, 2, 16] (k, s, ch_out)
      - [-1, 1, EConv, [3, 2, 16]] # k=3, s=2, ch_out=16
      - [-1, 1, Conv, [32, 3, 2]] # ch_out=32, k=3, s=2
      # Original: [64, 3, 2] (ch_out, k, s)
      # Corrected: [3, 2, 64] (k, s, ch_out)
      - [-1, 1, EConv, [3, 2, 64]] # k=3, s=2, ch_out=64
    
    head:
      # This is the last convolutional layer before the Detect head.
      # Its output will be fed directly to the Detect layer.
      - [-1, 1, Conv, [64, 3, 1]] # ch_out=64, k=3, s=1
      # The Detect layer will now take input from the immediately preceding Conv layer (layer 3)
      # and handle the final prediction convolutions internally.
      - [[-1], 1, Detect, [nc]] # Detect layer takes input from the immediately preceding layer
    
    ```
    
- create econv_test.ipynb
    
    ```python
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import EConv  # Import your custom layer directly
    import traceback
    ```
    
    ```python
    # 1. Test standalone layer
    print("Testing standalone EConv:")
    try:
        # Create EConv layer directly
        layer = EConv(c1=3, c2=16, k=3, s=1)
        print(f"Layer created: {layer}")
    
        # Test forward pass
        x = torch.randn(1, 3, 64, 64)
        out = layer(x)
        print(f"✓ Forward pass successful! Input shape: {x.shape}")
        print(f"✓ Output shape: {out.shape}")
    
        # Verify parameters
        params = sum(p.numel() for p in layer.parameters())
        print(f"✓ Parameters count: {params:,} (expected ~1,000)")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
    ```
    
    ```python
    print("\nTesting classification model:")
    try:
        model = YOLO('yolov8n_econv_classify.yaml', task='classify')
        print("Model loaded successfully!")
    
        # Verify parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
        # Test forward pass directly
        dummy = torch.rand(1, 3, 64, 64)  # [0-1] range
    
        # Use model's forward method
        with torch.no_grad():
            output = model.model(dummy)
    
        print(f"Output shape: {output.shape}")  # Should be [1, 10]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    ```
    
    ```python
    print(model.model)
    ```
    
    ```python
    print("\nTesting detection model:")
    try:
        # Load model with detection task
        model = YOLO('yolov8n_econv_detect.yaml', task='detect')
        print("Model loaded successfully!")
    
        # Verify parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
        # Test forward pass
        dummy = torch.rand(1, 3, 640, 640)  # Standard detection input size
    
        # Use model's forward method
        with torch.no_grad():
            output = model.model(dummy)
    
        # Output should be a tuple of feature maps
        print("Output shapes:")
        for i, x in enumerate(output):
            if isinstance(x, torch.Tensor):
                print(f"  Output {i+1}: {x.shape}")
            else:
                print(f"  Output {i+1}: {type(x)}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    ```