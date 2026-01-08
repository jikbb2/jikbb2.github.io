---
title: "[Review]NeRF: Neural Radiance Fields for View Synthesis"
date: 2025-08-08 18:41:11 +0900
categories: [AI, 3D]
tags: [review]     # TAG names should always be lowercase
math: true
description: This is review for the Neural Radiance Fields for View Synthesis

---
> **ECCV 2020**
> 
> **NeRF: Neural Radiance Fields for View Synthesis(19 Mar 2020)**
> 
> Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
> *UC Berkeley \| Google Research \| UC San Diego*
>
> [![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)]( https://arxiv.org/abs/2003.08934 ) [![Page](https://img.shields.io/badge/Project-Page-blue)]( https://www.matthewtancik.com/nerf ) [![Github](https://img.shields.io/badge/Github-Code-181717?logo=github)]( https://github.com/bmild/nerf ) 

 <br>
 ![Desktop View](/assets/img/nerf/nerf_info.webp){: .normal}
&nbsp; 이번 포스팅에서는 Inverse Rendering 분야에 큰 영향을 준 **NeRF**에 대해서 리뷰를 해보려고 합니다. **Neural Radiance Fields for View Synthesis**는 이름에서 알 수 있듯이 **"View Synthesis"** 즉, 다른 시점에서 촬영된 몇장의 이미지를 이용해 특정 시점에서의 장면을 복원하는 기술에 대한 내용입니다. 즉, 공간에서 촬영된 사진들을 가지고 3D로 복원하는 것을 의미합니다. 


 ![Desktop View](/assets/img/nerf/3d_modeling.webp){: width="350" .left} 

 &nbsp; NeRF는 암시적인 표현 방법으로 보통 알고 있는 렌더링 방식과는 다릅니다. 3D를 렌더링하기 위한 대표적인 방법으로는 Voxel, PointCloud, Mesh 등이 있습니다. 이 방식들은 우리가 흔히 알고 있는 렌더링 방법으로써 모두 **명시적인 표현 방법**입니다. 왜냐하면 최종적으로 렌더링되는 3D의 한 점이 대응하는 좌표 값을 직접적으로(명시적으로) 표현할 수 있기 때문입니다. 하지만 NeRF는 이와는 다른 렌더링 방법을 사용하고 있습니다. 저자는 연속적인 View Synthesis를 위해 미분 가능한 함수를 통해서 렌더링할 것을 제안하고 있는데, 이것은 렌더링 된 3D의 한 점을 직접적으로 표현할 수 없다는 것을 의미합니다. 따라서 이는 **암시적인 표현 방법**입니다. 다음 챕터부터 자세하게 알아보도록 하겠습니다.

 
## 1. Introduction

![Desktop View](/assets/img/nerf/camera_ray.svg){: width="350" }

현실 세계를 컴퓨터 그래픽으로 구현하는 가장 직관적인 방법은 Ray Tracing입니다. 물리적으로는 광원에서 출발한 빛이 물체에 부딪히고 반사되어 우리 눈(카메라)에 들어옵니다. 하지만 사방으로 흩어지는 무수히 많은 빛을 모두 계산하는 것은 불가능에 가깝습니다.그래서 우리는 **'역발상'**을 합니다. 카메라 렌즈로 들어오지 않는 빛은 계산할 필요가 없으니까요.즉, 카메라에서 가상의 시선(Ray)을 쏘아 보낸 뒤, 그 광선이 물체와 부딪히는 지점의 색상을 가져오는 방식을 사용합니다. NeRF는 바로 이 View Ray(Camera Ray) 위에서 일어나는 일을 모델링하는 기술입니다.


NeRF의 핵심은 이 가상의 광선이 지나가는 3차원 공간을 어떻게 표현하느냐에 있습니다. 저자들은 이를 위해 **Radiance**라는 개념을 가져옵니다. 간단히 말해, Radiance는 **"어떤 위치에서 특정 방향으로 뿜어져 나오는 빛의 색상과 세기"**입니다.우리가 물체를 볼 때, 보는 각도에 따라 빛 반사가 달라져 색이 다르게 보이는 것을 떠올리면 이해가 쉽습니다.따라서 NeRF는 3D 공간을 하나의 거대한 함수 $F_\Theta$ 로 정의합니다. 이 함수는 다음과 같은 입력을 받아 출력을 내놓습니다.

- 입력 (5D): 공간상의 위치 $X(x, y, z)$ + 바라보는 방향 $d(\theta, \phi)$
- 출력: 해당 지점의 **색상($\mathbf{c}$)**과 그 위치에 물체가 존재할 확률인 **밀도($\sigma$)**

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

이 함수를 근사(Approximation)하기 위해 Deep Fully-Connected MLP가 사용됩니다. 즉, 인공지능 모델이 "이 좌표, 이 각도에서는 빨간색 빛이 진하게 보여!"라고 알려주는 셈입니다. 이 설명은 아래의 챕터에서 더 자세하게 다루는 것으로 하고, 전반적인 NeRF의 파이프라인을 확인해보겠습니다. 
<br>
### NeRF Pipeline
![Desktop View](/assets/img/nerf/nerf_pipeline.webp)

여기서는 전체 파이프라인을 완벽히 이해하기보단 아래에서 하나씩 살펴볼 때, 어디 부분을 진행하고 있는가를 알기 위한 정도로 봐주시면 좋을 것 같습니다. 물론 논문에서도 이 위치에 간단한 파이프라인이 적혀있지만 개인적으로도 깊이 공부하다보면 어디를 보고 있었던건가 하는 생각이 종종 들 때가 있는 것 같아서 숲을 보고 나무를 보는 방식을 좋은 것 같아 먼저 정리해보려고 합니다. 


NeRF는 COLMAP과 같은 SfM(Structure-from-Motion) 알고리즘을 통해 미리 이미지들의 카메라 파이미터를 확보한 상태에서 시작합니다. SfM은 다각도에서 촬영된 이미지들을 통해서 Sparse 한 point cloud 를 생성하고, 촬영된 위치(카메라 파라미터)를 계산하는 기술입니다. 구체적으로 SfM을 다루기에는 내용이 많아서 다른 포스팅에서 깊게 다루도록 하겠습니다.  

**1. Image based Ray Generation**
<br>SfM을 통해 얻은 카메라 파라미터를 통해서 카메라의 각 픽셀마다 3D 공간으로 뻗어 나가는 Ray를 생성할 수 있습니다.


**2. Sampling**

![Desktop View](/assets/img/nerf/models_nerf-pipeline-sampler-light.png)

<br> Uniform / PDF Sampler 에서 3D 공간으로 뻗어 나가는 Ray를 토막 냅니다. 쪼개진 토막(sample) 단위로 밀도와 색상을 구하고 최종 새상을 구하기 위함입니다. 


**3. NeRF Field**

![Desktop View](/assets/img/nerf/nerf_pipeline_3.webp)

<br>View Ray는 텅 빈 허공을 지나갈 수도, 물체를 뚫고 지나갈 수도 있습니다. NeRF는 Ray 위의 여러 지점을 샘플링한 뒤, 앞서 정의한 MLP에게 물어봅니다."여기($(x,y,z)$)에 뭔가가 있니($\sigma$)? 있다면 내가 보는 방향($\theta, \phi$)에서 무슨 색($c$)이니?" NeRF Field에서는 이렇게 각각의 Sample 들을 MLP에 Query 하는 과정을 진행하게 됩니다. 이 과정을 통해 View Ray가 입력 위치까지 도달하면서 축적한 각 sample 위치에서의 방출된 색상을 얻을 수 있습니다. 


**4. Volume Rendering**

![Desktop View](/assets/img/nerf/nerf_pipeline_2.webp)

<br> NeRF Field에서 얻은 답변들을 적분하면, 최종적으로 카메라 픽셀에 맺히는 색상을 얻게 됩니다. 이 과정이 미분 가능하기 때문에 입력 이미지와 픽셀단위 비교를 통해서 최적화를 진행할 수 있습니다. 최적화가 끝나면 학습된 MLP를 가지고 최종 결과물을 렌더링할 수 있습니다. 


### Contribution
본문에서는 다음 내용을 contribution으로 소개하고 있습니다. 
- 복잡한 기하학적 구조에 대해서도 5D 뉴럴 래디언스 필드로 표현할 수 있는 접근 방식 제안.
- 전통적인 볼륨 렌더링 기법에 기반, 그리고 미분 가능한 렌더링 함수를 사용한 최적화 방식 제안.
- 위치 인코딩을 통한 5D의 성공적인 최적화, 그리고 고주파의 장면 표현 가능. 

## 2. Related Work
본 논문은 크게 **Neural 3D Shape Representations**과 **View Synthesis** 분야의 기존 연구들과 밀접한 관련이 있습니다. 이 섹션에서는 각 분야의 기존 접근 방식들이 가진 한계점과, NeRF가 이를 어떻게 극복했는지 살펴보겠습니다.
### Neural 3D Shape Representations
최근 연구들은 3D 좌표$(x,y,z)$를 입력받아 **SDF**나 **Occupancy Field**로 매핑하는 딥러닝 모델을 통해, 3D 형태를 **암시적으로 표현**하는 방법을 탐구해 왔습니다.

암시적인 표현의 초기 연구방법은 3D 좌표를 신경망에 통과시켜 레벨셋으로 형상을 표현했습니다. 하지만 모델 학습을 위해 **Ground Truth 3D Geometry**가 필요하다는 큰 제약이 있었습니다.
<br>Niemeyer et al., Sitzmann et al.의 후속 연구는 실제 3D 데이터 없이 2D 이미지만으로 학습이 가능하도록 **미분 가능한 렌더링 함수(Differentiable Rendering Function)**를 도입했습니다.


하지만 이 기법들은 기하학적 복잡도가 낮은 단순한 형태에 국한되어 있습니다. 복잡한 장면을 표현하려 할 경우는 결과물이 지나치게 뭉개지는 현상이 발생하여 고해상도의 기하학 구조를 표현하는 데 아쉬움이 있었습니다.

**NeRF의 해결책**
<br>저자는 3D 좌표뿐만 아니라 View Direction 정보까지 포함한 5D Radiance Field를 최적화하는 전략을 제안합니다. 이를 통해 단순한 형태를 넘어, 복잡한 장면에서도 고해상도의 기하학 구조와 사실적인 외형을 표현할 수 있음을 증명했습니다.

### View synthesis
View synthesis 분야에서는 관찰된 이미지로부터 기하학적 정보와 외형을 예측하는 방식들이 연구되어 왔습니다.

**Mesh-based approach**
<br>전통적으로 많이 사용되는 방식으로, Mesh를 최적화하여 장면을 표현합니다.
이 기법은 Gradient Descent를 통한 최적화 과정에서 Local Minima에 빠지기 쉽습니다.
또한 기존의 Mesh 방식은 자르거나 구멍 뚫기가 어렵기 때문에 최적화를 시작하기 전 고정된 Topology의 템플릿 mesh가 필요합니다. 이는 제약이 없는 실제 장면 데이터에는 적용하기 어렵게 만듭니다.

**Volumetric based approach**
<br>Voxel 그리드와 같은 볼륨 표현 방식을 사용하여 복잡한 형태와 재질을 표현합니다.
<br>이 기법의 장점은 mesh 기반 방식보다 시각적인 Artifact가 적고, Gradient 기반 최적화에 유리합니다.
하지만 공간을 Discrete하게 샘플링해야 하므로, 해상도를 높이려 할수록 메모리와 연산 비용이 기하급수적으로 증가합니다. 결국 고해상도 이미지를 렌더링하기에는 확장성이 떨어지는 한계가 존재합니다. 

쉽게 말하면, 3D 공간을 조사할 때 항상 고정된 눈금자를 사용하는 것과 같습니다. 매번 정확히 1cm, 2cm, 3cm 지점만 확인한다면, 1.5cm 지점에 무엇이 있는지는 영원히 알 수 없습니다.

이렇게 되면 MLP는 확인했던 그 특정 지점들의 색상만 암기하게 되고, 점과 점 사이의 연속적인 정보는 학습하지 못합니다. 결과적으로 마치 해상도가 낮은 모자이크 그림처럼, 정해진 격자 위치 외에는 표현할 수 없는 **'해상도 제한'**이 발생하게 됩니다.

**NeRF의 해결책**
<br>NeRF는 이산적인 복셀 그리드 대신, Fully-Connected MLP의 파라미터 자체에 연속적인 볼륨 정보를 인코딩하는 방식을 택했습니다. 이를 통해 이산 샘플링의 한계를 극복하여 훨씬 적은 저장 공간으로도 고해상도의 사실적인 렌더링이 가능해졌습니다.


## 3. Neural Radiance Field Scene Representation

![Desktop View](/assets/img/nerf/models_nerf-field-light.png){: width="350"}


NeRF에서는 연속적인 장면을 5D 벡터를 입력으로 하는 함수로 표현하며 파이프라인 그림의 NeRF Field 과정에서 동작합니다. 위치벡터 X($x, y, z$)와, 방향벡터 d($\theta, \phi$)를 입력으로 방출 색상 $C$ 와 볼륨 밀도 $\sigma$를 출력값을 얻게 됩니다. 

여기서 색상 $C$ 는 RGB 로 표현이 되며 시점에 따라 달라지는 방출 색상(emitted radiance)이고, 밀도 $\sigma$ 는 불투명도를 의미합니다. View ray는 불투명도가 1이 될때까지 방출되는 radiance를 축적하는데, 반대로 말하면 축적될 radiance의 양을 $\sigma$로 제어할 수 있다는 의미가 됩니다.

이 함수(NeRF Field)는 convolutional layer가 없는 Fully connected MLP로 표현되며 저자는 여러 방향에서 동일한 지점을 바라봤을 때 일관성을 유지할 수 있도록 $\sigma$ 는 위치벡터에만 영향을 받도록 하고, $C$는 방향 벡터와 위치벡터 모두에 의해 영향을 받을 수 있도록 설계하였습니다. 

![Desktop View](/assets/img/nerf/mlp_structure.webp)

$$F_\Theta : (x, d) \rightarrow (c, \sigma)$$

구체적으로 $F_\theta$는 먼저 위치 벡터 $X$를 8개의 Fully Connected layer로 처리해서 $\sigma$와 256차원의 feature 벡터를 출력합니다. 그리고 이 feature 벡터를 방향 벡터 $d$와 결합해서 128 채널의 추가적인 layer에 통과시켜서 **view-dependent**한 $C$를 출력합니다. 

위 그림은 NeRF에서 사용된 MLP 구조를 나타냅니다. 파란색 상자는 hidden layer를 표현하며, 초록색 상자는 input을, 빨간색 상자는 output을 표현합니다. 또한 검정색 실선 화살표는 ReLU activation을, 검정색 점선 화살표는 sigmoid activation을, 주황색 화살표는 activation이 없음을 나타냅니다.

위 그림에서 주황색 화살표 다음의 256차원이 8개의 layer를 거친 feature vector 이고, $\gamma$ 함수는 positional encoding을 의미합니다. Positional Encoding에 대해서는 **5. Optimizing a Neural Radiance Field** 에서 자세하게 다루도록 하겠습니다.

![img-descrption](/assets/img/nerf/non-lambertian_effects.png)
_In (a) and (b), we show the appearance of two fixed 3D points from two different camera positions_

위 그림에서 view-dependent한 색상이 어떻게 표현되는지 알 수 있습니다. 그리고 다음 그림에서 view-dependent 없이 훈련된 모델에서는 하이라이트를 표현하는 데 어려움을 겪는 것을 알 수 있습니다.
 
![img-descrption](/assets/img/nerf/no_viewdependent.png)
_Removing view dependence prevents the model from recreating the specular reflection on the bulldozer tread._


## 4. Volume Rendering with Radiance Fields
![Desktop View](/assets/img/nerf/nerf_pipeline_3.webp)

앞선 챕터에서는 위 그림의 NeRF Field 단계에 대해 살펴보았습니다. 과정을 요약하면 다음과 같습니다.우리는 기본적으로 입력 이미지와 그에 대응하는 카메라의 위치 및 방향 정보를 가지고 있습니다. 이 정보를 활용하면 카메라 원점에서 이미지의 특정 픽셀을 관통하는 **카메라 광선**을 정의할 수 있습니다.이후 각 광선 위에서 여러 지점(Point)을 샘플링하고, 이를 NeRF Field에 통과시킴으로써 각 샘플 포인트에 해당하는 색상($c$)과 밀도($\sigma$) 정보를 얻어낼 수 있습니다.

하지만 이 과정을 거치게 되면 최종 색상이 아닌 각각의 샘플에서의 밀도와 색상을 얻은 것 뿐입니다. 따라서 최종적인 색상을 구하기 위한 렌더링 과정이 필요합니다. 

![Desktop View](/assets/img/nerf/nerf_pipeline_2.webp)

이를 위해 고전적인 볼륨 렌더링 원리를 사용하며, 샘플들의 색상과 밀도를 가지고 적분을 함으로써 최종적으로 렌더링 할 색상을 구하고 GT와의 손실 값을 통해서 학습을 진행할 수 있습니다. 이것이 이 챕터에서 진행하고자 하는 내용이며, 위 단계에서 진행됩니다.

![Desktop View](/assets/img/nerf/volume_render.png)

가상의 Volume space를 가상의 카메라 광선이 바라보는 이미지를 뚫고 지나가는 것을 생각했을 때, 볼륨 밀도 $\sigma$는 카메라 광선이 위치 $X$의 입자에서 종료될 확률로 해석될 수 있습니다. 위 이미지와 함께 이해해보겠습니다.

위 그림(a)을 보면 두개의 이미지, 두개의 카메라 광선을 보여주고 있습니다. 카메라 광선은 각 이미지의 특정 픽셀을 관통하여 지나가며 선 위의 점들은 샘플링된 지점을 의미합니다. 그리고 각 샘플링된 지점이 $F_\theta$를 지나 그림 (b)를 얻을 수 있습니다.

알다시피 그림(b)에서 각 샘플링된 지점에서는 해당하는 밀도와 색상을 가지고 있습니다. 이것을 가지고 각 광선별로 근경계에서 원경계까지 밀도 값을 표현한 것이 그림(c)가 됩니다. 여기서는 밀도가 불투명도와 동일하게 작동하기 때문에 밀도가 1이 되는 지점까지의 색상을 구하는 것이 이번 챕터의 목표가 될 것입니다. 

 밀도가 1이 되는 지점까지의 색상을 $C(r)$라고 했을때, 카메라 위치 $o$와 방향벡터 $d$를 가진 카메라 광선 $r(t)=o+td$ 의 기댓값 색상 $C(r)$는 다음과 같습니다. 여기에서 $t_n$ 은 Volume space 의 근경계, $t_f$는 원경계에서의 위치를 의미합니다.

$$C(r)=\int_{t_n}^{t_f}T(t) \cdot \sigma(r(t)) \cdot c(r(t),d)dt$$

위 식을 각 항마다 이해하기 쉽게 다음과 같이 표현할 수 있습니다.

| [기대 색상] = [해당 지점까지 빛이 도달할 확률] × [해당 지점에서 빛이 멈출 확률] × [해당 지점의 색상]       

$$T(t)=\exp(- \int ^t_{t_n}\sigma(r(s))ds)$$


**[해당 지점까지 빛이 도달할 확률]** 로 표현했던 함수 $T(t)$는 $t_n$부터 $t$ 까지 광선에 의한 **누적 투과율**, 즉 광선이 다른 입자와 부딪히지 않고 $t_n$부터 $t$ 까지 진행할 확률을 나타냅니다. 

여기서 $\exp(-x)$는 **밀도를 가진 입자에 의해 감소**하는 빛의 투과량을 **0과 1 사이의 확률**로 나타내기 위해 사용합니다. 

이제 연속적인 적분 식을 컴퓨터가 계산할 수 있도록 이산적인 합으로 바꿔야 합니다. 앞서 본 적분 식은 이론적으로 완벽하지만, 컴퓨터에서는 무한히 작은 구간을 적분할 수 없습니다. 따라서 우리는 광선을 $N$개의 작은 구간으로 나누어 계산하는 **구적법**을 사용합니다.

$$\hat C(r)=\sum_{i=1}^N T_i \cdot \alpha_i \cdot c_i$$

여기서 $\alpha_i$는 $i$번째 구간의 **불투명도**를 의미하며 다음과 같이 정의됩니다. 

$$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$$

위 수식들이 의미하는 바를 하나씩 뜯어보면 매우 직관적입니다.
NeRF 네트워크는 특정 지점의 **밀도($\sigma_i$)**를 뱉어냅니다. 하지만 렌더링을 하려면 **그 구간이 얼마나 불투명한지**를 알아야 합니다. 

- **$\delta_i$** : 여기서 $\delta_i=t_{i+1} − t_i$는 $i$번째 샘플과 $i+1$번째 샘플 사이의 거리입니다. 

- **$\sigma_i \delta_i$** : 밀도($\sigma$)가 높을수록, 그리고 그 구간의 길이($\delta$)가 길수록 빛이 통과하기 힘듭니다. 

- **$\alpha_i$** : 위 식 $1 - \exp(-\sigma_i \delta_i)$를 통해 0에서 1 사이의 값으로 변환됩니다. 이것이 우리가 흔히 아는 그래픽스의 Alpha 값입니다.

$$T_i= \exp( - \sum_{j=1}^{i - 1}\sigma_j\delta_j) = \prod_{j=1}^{i-1} (1-\alpha_j)$$

$T_i$는 광선이 카메라에서 출발해 $i$번째 구간에 도달할 때까지, 앞선 장애물($1 \sim i-1$ 구간)들에 의해 가려지지 않고 살아남은 빛의 비율입니다. 이전 설명에서의 **누적 투과율**과 동일하며, 앞쪽에서 불투명한 물체를 만났다면 $T_i$는 0에 가까워져, 뒤쪽 색상($c_i$)은 결과에 거의 반영되지 않게 됩니다.

결국 최종 색상 $\hat C(r)$은 각 구간의 색상($c_i$)에 **해당 구간의 불투명도**와 **그 지점까지 빛이 도달할 확률**을 가중치로 곱해서 모두 더한 것입니다.이 과정은 컴퓨터 그래픽스에서 전통적으로 사용해 온 Alpha Compositing 방식과 수학적으로 정확히 일치합니다.

 중요한 점은, 이 모든 과정이 미분 가능하도록 설계되어 있어 최적화가 가능하다는 것입니다.
 
 또한 **2. Related Work**의 **Volumetric based approach**에서 살펴보았던 복셀 그리드를 렌더링하는 데 일반적으로 사용하던 구적법의 문제점은 이산적인 위치에서만 계산되도록 하므로 해상도를 제한하는 것이 문제로 제시되었었습니다. 

$$t_i\sim u[t_n+\frac{(i − 1)}{N}(t_f − t_n),t_n+\frac{i}{N}(t_f − t_n)]$$

따라서 이를 위해 Stratified sampling 접근 방식을 사용하여 해결하고자 합니다. 이 방식은 $[t_n,t_f]$ 구간을 N개의 균일한 간격으로 분할하되, 각 구간 내에서 균등분포에 따라 무작위로 하나의 샘플을 추출합니다.

비록 적분을 위해 이산적인 샘플 집합을 사용하지만, 계층적 샘플링은 샘플링 위치가 계속 바뀌게 되면서 반복 학습 과정 동안 MLP가 연속적인 위치에서 평가되도록 하므로 결과적으로 연속적인 장면 표현을 나타낼 수 있게 합니다.


## 5. Optimizing a Neural Radiance Field

이제 NeRF에서 사용한 최적화를 위한 추가적인 아이디어를 살펴보겠습니다.

### Positional encoding

![Desktop View](/assets/img/nerf/no_pe.png)

NeRF 모델의 입력값인 위치($x$)와 방향($d$)을 있는 그대로 MLP에 넣으면 흐릿하고 뭉개진 이미지가 나옵니다. 이는 딥러닝의 MLP가 본질적으로 저주파 편향(Low-frequency Bias) 문제를 가지고 있기 때문입니다. 쉽게 말해, MLP는 급격하게 변하는 복잡한 디테일(고주파)보다는 부드럽게 변하는 단순한 신호(저주파)를 더 잘 학습하는 경향이 있습니다.이 문제를 해결하기 위해 NeRF는 입력 데이터를 고차원 공간으로 매핑하는 Positional Encoding 기법을 도입했습니다.

입력값 $p$를 다양한 주파수의 사인, 코사인 함수에 통과시켜 더 높은 차원의 벡터로 변환합니다.

$$\gamma(p) = \sin
(2^0 \pi p), \cos (2^0 \pi p), \cdots , \sin(2^{L − 1}\pi p), \cos(2^{L − 1}\pi p)$$ 

이 함수를 거치면 단순했던 $x, y, z$ 좌표값이 미세한 차이까지 구별할 수 있는 고차원의 정보로 확장됩니다. 수학적으로는 MLP 함수 $F'_\Theta$와 인코딩 함수 $\gamma$의 합성 함수 형태로 전체 네트워크를 표현할 수 있습니다.

$$F_\Theta = F'_\Theta \circ \gamma$$

본 실험에서는 $\gamma(X)$에 대해 $L = 10$, $\gamma(d)$에 대해 $L = 4$ 로 설정하여 진행했습니다.

**위치 벡터에 비교적 높은 L을 적용하는 이유**
<br>장면의 기하학적 구조나 질감은 아주 미세한 위치 변화에도 급격하게 바뀔 수 있기 때문에 더 큰 주파수 사용. 
<br>ex) 머리카락 한 올, 나뭇결 무늬

**방향 벡터에 비교적 낮은 L을 적용하는 이유**
<br>시점 의존적 효과는 일반적으로 보는 각도에 따라 부드럽게 변하기 때문에 비교적 낮은 주파수 사용. 
<br>ex) 하이라이트, 반사

### Hierarchical volume sampling

지금까지의 방식으로는 ray 당 $N$개의 질의 지점을 갖게 되는데, 이 방식으로 뉴럴 래디언스 필드 네트워크를 빽빽하게 평가하는 전략은 비효율적입니다. 최종적으로 렌더링된 이미지에 기여하지 않는 빈 공간이나 가려진 영역도 반복적으로 샘플링 되기 때문입니다.

그래서 Coarse, Fine 두번으로 나누어 계층적으로 샘플링하는 전략을 취합니다. **Coarse Network**에서는 표면이 존재할 것 같은 대략적인 위치를 제안하는 역할을 하며, **Fine Network**에서는 해당 제안을 바탕으로 효과적인 샘플링을 진행하게 됩니다.

**Coarse Network**

![Desktop View](/assets/img/nerf/coarse_network.webp)

$$\hat C_c(r)= \sum _{i=1}^{N_c}w_ic_i$$

먼저, 전체 공간을 대략적으로 탐색하는 단계입니다. 앞서 설명한 Stratified Sampling을 사용하여 광선 위에서 $N_c$개의 샘플을 균일하게 추출하고, Coarse Network를 통해 각 지점의 색상과 밀도를 계산합니다.

여기서 계산된 가중치 $w_i=T_i(1− \exp(− \sigma_i\delta_i))$는 해당 지점이 최종 색상 결정에 얼마나 기여하는지를 나타냅니다. 즉, 물체가 있어서 밀도가 높은 곳은 가중치가 크고, 빈 공간은 가중치가 작습니다.

이 $w_i$를 Normalization하면 광선을 따라 분포하는 **확률 밀도 함수(PDF)**를 얻을 수 있고, 이것을 Fine Network에서 더 효율적으로 샘플링을 하기 위해 사용됩니다.

**Fine Network**

![Desktop View](/assets/img/nerf/fine_network.webp)

이 단계에서는 Inverse Transform Sampling 기법을 사용합니다. PDF를 적분하여 누적 분포 함수(CDF)를 만들고, 이를 역으로 추적하여 물체가 있을 확률이 높은 구간에 더 많은 샘플을 집중적으로 배치합니다.

덕분에 빈 공간에 낭비되는 연산을 줄이고, 물체의 표면처럼 중요한 부분은 훨씬 정교하게 표현할 수 있습니다.

### Implementation details

$$L = \sum _{r\in R}[\left\| \hat C_c(r) - C(r) \right\|^2_2 + \left\| \hat C_f(r) - C(r) \right\|^2_2]$$

여기서 $\mathcal{R}$은 각 배치의 광선 집합이며, $C(r)$, $\hat C_c(r)$, $\hat{C}_f(r)$은 각각 광선 r에 대한 실제 값, coarse 볼륨 예측, fine 볼륨 예측 RGB 색상입니다. 

최종 렌더링은 Fine Network에서 계산된 값($\hat{C}_f(r)$)을 통해 나오지만, coarse 네트워크의 가중치 분포가 Fine Network의 샘플 할당에 사용되므로 Coarse Network의 손실도 최소화하여 정확도를 높일 수 있도록 합니다.

## 6. Additional Implementation Details

### Volume Bounds

- **합성 이미지 실험의 경우**, 저희는 장면이 원점을 중심으로 하는 변의 길이가 2인 정육면체 내에 있도록 크기를 조절하고, 이 **경계 볼륨(bounding volume)** 내에서만 표현에 질의합니다.

- **실제 이미지 데이터셋의 경우**, 콘텐츠가 가장 가까운 지점부터 무한대까지 어디에나 존재할 수 있으므로, 저희는 정규화된 장치 좌표계(NDC)를 사용하여 이러한 점들의 깊이 범위를 `[-1, 1]`로 매핑합니다. 이 변환은 모든 광선 원점을 장면의 근경 평면으로 이동시키고, 카메라의 원근 광선을 변환된 볼륨 내의 평행 광선으로 매핑하며, 미터 단위 깊이 대신 시차(역깊이)를 사용합니다. 결과적으로 모든 좌표는 유한한 경계를 갖게 됩니다.

 **NDC space dervation**

![Desktop View](/assets/img/nerf/ndc_dervation.webp){: width="350" }

만약 깊이를 기준으로 균일하게 sampling 한다면 먼 곳에서는 장면이 거의 변하지 않는데도 촘촘하게 sampling하여 계산자원을 낭비하게 됩니다.

이 문제를 해결하기 위해 NDC 공간으로 3D 공간을 의도적으로 왜곡하여 sampling에 반영합니다.
시차는 거리에 반비례하는 $1/z$ 로 표현 가능하며 NDC공간의 새로운 $z$축 좌표는 시차에 비례하게 됩니다.

### MLP Structure

![Desktop View](/assets/img/nerf/mlp_structure.webp)

 챕터 3. Neural Radiance Field Scene Representation 에서 살펴보았던 MLP 그림을 다시 보도록 하겠습니다. $\gamma$는 positional encoding 함수를 나타낸 다고 했습니다. 

또한 중요한 것은 3D point의 $x, d$가 MLP에 입력되는 타이밍과, 색상과 밀도의 출력 타이밍입니다. $x$만 갖고 layer를 투과하여 밀도가 예측되며, 밀도가 출력될 때 $d$ 정보까지 추가적으로 입력되어 색상이 예측됩니다.

 이는 NeRF의 밀도가 점의 위치에 따라 결정되고 색상은 점의 위치와 바라보는 방향에 따라 결정된다는 성질을 반영하고 있습니다.
 
그리고 5번째 hidden layer에서 skip connection을 위해 한번 더 $x$를 대입합니다.

## 7. Results

그래서 NeRF가 실제로 얼마나 뛰어난 성능을 보여주는지, 그리고 기존 방법론들과 비교했을 때 어떤 장단점을 가지는지 살펴보겠습니다. 또한, Ablation Study를 통해 NeRF의 핵심 아이디어들이 성능에 어떤 영향을 미치는지 분석합니다.

### Datasets
저자는 모델 검증을 위해 '합성 이미지'와 '실제 이미지' 두 가지 환경 모두에서 실험을 진행했습니다. 합성 이미지를 데이터 셋으로 사용한 이유는 통제된 환경에서 정확한 성능 측정을 위해 사용할 수 있기 때문입니다.

**합성 이미지 데이터셋**

- DeepVoxels 데이터셋: 기하학적 구조가 단순하고, Lambertian 표면을 가진 4개의 객체입니다.

- 자체 제작 데이터셋: NeRF의 강점을 보여주기 위해 직접 제작한 데이터셋입니다. 복잡한 기하학적 구조와 Non-Lambertian 재질을 가진 8개의 객체이며, 이 데이터셋은 View-dependent effects를 확인할 수 있습니다. 

**실제 이미지 데이터 셋**

- Real Forward-Facing: 휴대폰으로 촬영한 8개의 실제 장면입니다. (5개는 LLFF 논문 데이터, 3개는 직접 촬영).대략 정면을 향하는(Forward-facing) 구도로 촬영되었습니다.

### Comparisons
NeRF의 성능을 입증하기 위해 당시의 View Synthesis 모델들과 비교했습니다. 여기서는 간단하게 언급하고 추후에 다른 포스팅으로 다루겠습니다.

- Neural Volumes (NV): 배경이 없는 객체 중심의 뷰 합성에 사용됩니다. $128^3$ 해상도의 복셀 그리드를 생성하고 레이 마칭을 수행합니다.
- Scene Representation Networks (SRN): 연속적인 장면을 표현하지만, 각 광선에 대해 단일 깊이와 색상만을 선택하여 렌더링합니다.
- Local Light Field Fusion (LLFF): 입력 뷰를 통해 다중 평면 이미지를 생성하고 이를 블렌딩하여 새로운 뷰를 만듭니다. 실사 이미지 처리에 강점이 있습니다.

### Discussion
NeRF는 정량적 수치(PSNR, SSIM, LPIPS)뿐만 아니라 시각적인 품질(정성적 평가)에서도 모든 비교 모델을 압도했다고 평가하고 있습니다. 정성적 평가를 확인해 보겠습니다. 

**기존 모델들의 한계점**

![Desktop View](/assets/img/nerf/comparison.png)

**SRN**
- 광선당 하나의 깊이/색상만 선택하므로 기하학적 구조와 텍스처가 뭉개져서 흐릿하게 표현됩니다.

**NV** 
- $128^3$이라는 복셀 그리드의 해상도 제한 때문에 고해상도 이미지에서 디테일을 표현하지 못합니다.

**LLFF**
- 서로 다른 뷰의 표현을 블렌딩하는 방식을 사용하기 때문에, 렌더링된 영상에서 물체가 겹쳐 보이거나 끊기는 등의 비일관성이 발생합니다.
- 입력 이미지의 카메라 간격이 넓으면 기하학적 구조 추정에 실패합니다.


### Ablation studies
마지막으로 그동안 확인해본 기술들이 성능에 영향을 미치는 정도를 확인해보겠습니다. 

![Desktop View](/assets/img/nerf/ablation.png)

- **No Positional Encoding 또는 PE**단계에서는 고주파 디테일(선명함)이 사라집니다. 1, 2번 행에서 성능 저하를 확인할 수 있습니다.
- **No View Dependence 또는 VD**는 input에서 $\theta, \phi$를 제거를 의미하고, 이는 Specular을 표현하지 못하고 Lambertian 재질처럼만 보이게 만듭니다.
- **No Hierarchical 또는 H**는 Coarse Layer 만 수행하는 것을 의미하고, 1, 4번 행에서 성능 저하를 확인할 수 있습니다.  
- 테이블의 5,6번은 입력 이미지를 100장에서 25장으로 줄여도, 100장을 사용한 기존 모델(NV, SRN, LLFF)보다 성능이 뛰어나다는 것을 보여줍니다. 
- 테이블의 7,8번은 주파수 설정의 중요성을 알려주고 있습니다. 위치 인코딩의 주파수 $L$이 너무 낮으면(5) 성능이 떨어집니다.하지만 $L$을 무작정 높인다고(10 $\rightarrow$ 15) 성능이 계속 오르지는 않습니다. 샘플링된 이미지의 최대 주파수(나이퀴스트 이론)를 초과하면 이득이 제한적이기 때문입니다.

## 8. Conclusion

NeRF가 단순히 이론적으로만 훌륭한 것이 아니라, **"고해상도", "적은 저장 용량", "복잡한 기하학/조명 표현"**이라는 난제들을 실제로 해결했음을 보여줍니다. 특히 기존의 이산적(Discrete) 표현방식(복셀, 메쉬)이 가진 한계를 연속적(Continuous) 함수 표현으로 극복했다는 점이 실험 결과에서 명확히 드러납니다.

또한 NeRF는 복잡한 실사 데이터를 MLP상의 weight을 통해서 적은 크기로 표현할 수 있다는 점과, 깊이 및 3D 정보가 필요하지 않다는 점, 그리고 렌더링 과정이 미분 가능하기 때문에 gradient 기반 optimaization을 사용할 수 있다는 점으로 혁신적인 모델로 평가받고 있습니다.

하지만 학습시간이 매우 느리며, 정적인 scene에 대해서만 표현이 가능합니다. 또한 학습을 위해 사진들과 카메라 파라미터 정보가 필요하다는 단점이 존재합니다.

## 9. Reference

{% linkpreview "https://docs.nerf.studio/nerfology/methods/nerf.html" %}
{% linkpreview "https://arxiv.org/abs/2003.08934" %}

