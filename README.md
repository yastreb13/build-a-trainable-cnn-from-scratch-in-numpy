# Собрал свёрточную сеть в стиле LeNet на чистом NumPy
- от устойчивого softmax и im2col-свёрток до тренировочного цикла с Adam. Реализованы все слои, градиенты и оптимизатор, сеть обучается на синтетических картинках. Код полностью рабочий, без фреймворков. Архитектура и вся числодробилка на NumPy — мои, а оформление проекта (README, html-визуализация и файл scaffold.py) сделал сайт, включая вёрстку сайта со схемой сети.
## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** argmax_rows
- [x] **2.** row_max
- [x] **3.** row_sum
- [x] **4.** exp_shifted
- [x] **5.** stable_softmax
- [x] **6.** one_hot
- [x] **7.** gather_true_class_probs
- [x] **8.** cross_entropy_loss
- [x] **9.** accuracy
- [x] **10.** he_std
- [x] **11.** he_init
- [x] **12.** init_zero_bias
- [x] **13.** pad_2d
- [x] **14.** output_spatial_size
- [x] **15.** im2col
- [x] **16.** col2im
- [x] **17.** conv2d_forward
- [x] **18.** conv2d_grad_input
- [x] **19.** conv2d_grad_weights
- [x] **20.** conv2d_grad_bias
- [x] **21.** conv2d_backward
- [x] **22.** maxpool2d_forward
- [x] **23.** scatter_grad_window
- [x] **24.** maxpool2d_backward
- [x] **25.** relu_forward
- [x] **26.** relu_backward
- [x] **27.** flatten_forward
- [x] **28.** flatten_backward
- [x] **29.** linear_forward
- [x] **30.** linear_grad_input
- [x] **31.** linear_grad_weights
- [x] **32.** linear_grad_bias
- [x] **33.** linear_backward
- [x] **34.** softmax_cross_entropy_forward
- [x] **35.** softmax_cross_entropy_backward
- [x] **36.** sgd_step
- [x] **37.** adam_update_m
- [x] **38.** adam_update_v
- [x] **39.** adam_bias_correct
- [x] **40.** adam_param_step
- [x] **41.** adam_step
- [x] **42.** init_conv_layer
- [x] **43.** init_linear_layer
- [x] **44.** init_lenet
- [x] **45.** forward_conv_block
- [x] **46.** forward_classifier_block
- [x] **47.** lenet_forward
- [x] **48.** backward_conv_block
- [x] **49.** backward_classifier_block
- [x] **50.** lenet_backward
- [x] **51.** lenet_predict
- [x] **52.** build_synthetic_image_dataset
- [x] **53.** shuffle_indices
- [x] **54.** train_test_split
- [x] **55.** iterate_minibatches
- [x] **56.** train_step
- [x] **57.** train_one_epoch
- [x] **58.** train_loop
- [x] **59.** evaluate

---

Built on Deep-ML.
