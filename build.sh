wget --no-check-certificate https://paddle-serving.bj.bcebos.com/pddet_demo/2.0/faster_rcnn_r50_fpn_1x_coco.tar
tar xf faster_rcnn_r50_fpn_1x_coco.tar
cp ../python_infer.py .
WITH_MKL=OFF
LIB_DIR=/root/paddle_inference
cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=infer \
  -DWITH_STATIC_LIB=OFF 

make infer 

