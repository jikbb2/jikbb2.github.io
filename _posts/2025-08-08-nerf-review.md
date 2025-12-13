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
<br> Uniform / PDF Sampler 에서 3D 공간으로 뻗어 나가는 Ray를 토막 냅니다. 쪼개진 토막(sample) 단위로 밀도와 색상을 구하고 최종 새상을 구하기 위함입니다. 


**3. NeRF Field**
<br>View Ray는 텅 빈 허공을 지나갈 수도, 물체를 뚫고 지나갈 수도 있습니다. NeRF는 Ray 위의 여러 지점을 샘플링한 뒤, 앞서 정의한 MLP에게 물어봅니다."여기($(x,y,z)$)에 뭔가가 있니($\sigma$)? 있다면 내가 보는 방향($\theta, \phi$)에서 무슨 색($c$)이니?" NeRF Field에서는 이렇게 각각의 Sample 들을 MLP에 Query 하는 과정을 진행하게 됩니다. 


**4. Volume Rendering**
<br> NeRF Field에서 얻은 답변들을 Ray의 경로를 따라 쭉 적분하면, 최종적으로 카메라 픽셀에 맺히는 색상을 얻게 됩니다. 이 과정이 미분 가능하기 때문에 입력 이미지와 픽셀단위 비교를 통해서 최적화를 진행할 수 있습니다. 최적화가 끝나면 학습된 MLP를 가지고 최종 결과물을 렌더링할 수 있습니다. 


### Contribution
본문에서는 다음 내용을 contribution으로 소개하고 있습니다. 
- 복잡한 기하학적 구조에 대해서도 5D 뉴럴 래디언스 필드로 표현할 수 있는 접근 방식 제안.
- 전통적인 볼륨 렌더링 기법에 기반, 그리고 미분 가능한 렌더링 함수를 사용한 최적화 방식 제안.
- 위치 인코딩을 통한 5D의 성공적인 최적화, 그리고 고주파의 장면 표현 가능. 

## 2. Related Work
본 논문은 크게 **Neural 3D Shape Representations**과 **View Synthesis** 분야의 기존 연구들과 밀접한 관련이 있습니다. 이 섹션에서는 각 분야의 기존 접근 방식들이 가진 한계점과, NeRF가 이를 어떻게 극복했는지 살펴보겠습니다.
### Neural 3D Shape Representations
최근 연구들은 3D 좌표$(x,y,z)$를 입력받아 **SDF(Signed Distance Function)**나 Occupancy Field로 매핑하는 딥러닝 모델을 통해, 3D 형태를 **암시적으로 표현**하는 방법을 탐구해 왔습니다.

암시적인 표현의 초기 연구방법은 3D 좌표를 신경망에 통과시켜 레벨셋으로 형상을 표현했습니다. 하지만 모델 학습을 위해 **Ground Truth 3D Geometry**가 필요하다는 큰 제약이 있었습니다.
<br>Niemeyer et al., Sitzmann et al.의 후속 연구는 실제 3D 데이터 없이 2D 이미지만으로 학습이 가능하도록 **미분 가능한 렌더링 함수(Differentiable Rendering Function)**를 도입했습니다.


하지만 이 기법들은 기하학적 복잡도가 낮은 단순한 형태에 국한되어 있습니다. 복잡한 장면을 표현하려 할 경우는 결과물이 지나치게 뭉개지는 현상이 발생하여 고해상도의 기하학 구조를 표현하는 데 아쉬움이 있었습니다.

**NeRF의 해결책**
<br>저자들은 3D 좌표뿐만 아니라 2D 시점(View Direction) 정보까지 포함한 5D Radiance Field를 최적화하는 전략을 제안합니다. 이를 통해 단순한 형태를 넘어, 복잡한 장면에서도 고해상도의 기하학 구조와 사실적인 외형을 표현할 수 있음을 증명했습니다.

### View synthesis
새로운 뷰를 합성하는 분야에서는 관찰된 이미지로부터 기하학적 정보와 외형을 예측하는 방식들이 연구되어 왔습니다.

**Mesh-based approach**
<br>전통적으로 많이 사용되는 방식으로, Mesh를 최적화하여 장면을 표현합니다.
이 기법은 Gradient Descent를 통한 최적화 과정에서 **Local Minima**에 빠지기 쉽습니다.
또한 최적화를 시작하기 전 고정된 Topology의 템플릿 mesh가 필요합니다. 이는 제약이 없는 실제 장면 데이터에는 적용하기 어렵게 만듭니다.

**Volumetric based approach**
<br>Voxel 그리드와 같은 볼륨 표현 방식을 사용하여 복잡한 형태와 재질을 표현합니다.
<br>이 기법의 장점은 mesh 기반 방식보다 시각적인 Artifact가 적고, Gradient 기반 최적화에 유리합니다.
하지만 공간을 Discrete하게 샘플링해야 하므로, 해상도를 높이려 할수록 메모리와 연산 비용이 기하급수적으로 증가합니다. 결국 고해상도 이미지를 렌더링하기에는 확장성이 떨어지는 한계가 존재합니다.

**NeRF의 해결책**
<br>NeRF는 이산적인 복셀 그리드 대신, Fully-Connected MLP의 파라미터 자체에 연속적인 볼륨 정보를 인코딩하는 방식을 택했습니다. 이를 통해 이산 샘플링의 한계를 극복하여 훨씬 적은 저장 공간으로도 고해상도의 사실적인 렌더링이 가능해졌습니다.


## 3. Neural Radiance Field Scene Representation

![Desktop View](/assets/img/nerf/neural_field.webp)

NeRF에서는 연속적인 장면을 5D 벡터를 입력으로 하는 함수로 표현하며 우측 그림의 파이프라인과정에서 동작합니다. 위치벡터 X($x, y, z$)와, 방향벡터 d($\theta, \phi$)를 입력으로 방출 색상 $C$ 와 볼륨 밀도 $\sigma$를 출력값을 얻게 됩니다.
## 4. Volume Rendering with Radiance Fields
![Desktop View](/assets/img/nerf/nerf_pipeline_2.webp)


Neural Radiance Fields는 한 장면을 공간상의 모든 지점에서의 볼륨 밀도와 방향성 방출 색상으로 표현합니다. 고전적인 볼륨 렌더링 원리를 사용하여 장면을 통과하는 모든 광선의 색상을 렌더링합니다.
![Desktop View](/assets/img/nerf/nerf_pipeline_3.webp)


가상의 Volume space를 가상 카메라 광선(ray)가 바라보는 이미지를 뚫고 지나가는 것을 생각했을 때, 볼륨 밀도 σ는 카메라 광선이 위치 x의 입자에서 종료될 확률로 해석될 수 있습니다. 카메라 위치 o 와 방향벡터 d를 가진 카메라 광선 r(t)=o+td의 기댓값 색상 C(r)는 다음과 같습니다:

$$C(r)=\int_{t_n}^{t_f}T(t)\sigma(r(t))c(r(t),d)dt,where T(t)=\exp(- \int ^t_{t_n}\sigma(r(s))ds)$$

| [기대 색상] = [해당 지점까지 빛이 도달할 확률] × [해당 지점에서 빛이 멈출 확률] × [해당 지점의 색상]

함수 $T(t)$는 $t_n$부터 $t$ 까지 광선에 의한 누적 투과율, 즉 광선이 다른 입자와 부딪히지 않고 $t_n$부터 $t$ 까지 진행할 확률을 나타냅니다.
Neural radiance Field 로부터 3D를 렌더링하는 것은 이미지의 각 픽셀을 통과하는 카메라 광선에 대해서 적분하는 것으로 얻을 수 있습니다.

$$ti\sim u[t_n+\frac{(i − 1)}{N}(t_f − t_n),t_n+\frac{i}{N}(t_f − t_n)]$$

구적법을 사용하여 이전의 연속적인 적분을 수치적으로 추정합니다. 저자는 Stratified sampling 접근 방식을 사용하여 $[t_n, t_f]$ 구간을 $N$개의 균일한 간격으로 분할합니다.

$$\hat C(r)=\sum_{i=1}^N T_i(1− \exp(− \sigma_i\delta_i))c_i,whereT_i= \exp( − \sum_{j=1}^{i − 1}\sigma_j\delta_j)$$

$δ_i = t_{i+1} — t_i$ 는 인접한 sample 사이의 거리입니다. $σδ$ 로 특정 지점이 아닌 하나의 sample 구간 내에서의 밀도를 의미하게됩니다. 위 식은 미분이 가능하며, 알파합성이 가능하다는 것을 보여주고 있습니다.



샘플링된 ray에 축적된 공간내의 색상들을 알파합성을 통해서 총 하나의 ray의 축적된 색을 표현할 수 있습니다.


## 5. Optimizing a Neural Radiance Field
![Desktop View](/assets/img/nerf/neural_field.webp)

위 볼륨 렌더링 공식을 통해서 결론적으로 우리가 공간에서 보고자 하는 색상을 계산할 수 있습니다. 이때 특정 샘플에 해당하는 밀도 값과 해당 지점의 색상을 최적화과정을 통해서 적합한 값을 추정하게 됩니다.


### Positional encoding
이 과정에서 NeRF에서는 positional encoding을 사용하여 최적화를 진행하였다고 합니다. 이는 MLP에서 저주파 편향 현상 문제를 가지고 있어 네트워크에 x와 d를 직접적으로 적용하면 발생하는 성능 저하문제를 해결합니다.

$$\gamma(p) = \sin
(2^0 \pi p), \cos (2^0 \pi p), \cdots , \sin(2^{L − 1}\pi p), \cos(2^{L − 1}\pi p)$$ 

γ는 입력값을 더 높은 차원의 공간으로 매핑하는 Positional encoding 함수입니다. 삼각함수를 사용하여 미세한 차이를 더 극명하게 드러나게함과 동시에 값의 정보는 손실되지 않도록 할 수 있습니다.

$F_Θ=F'_Θ∘\gamma$처럼 MLP 함수 $F’_Θ$ 와 인코딩된 $\gamma$ 함수의 합성곱 형태로 MLP를 표현할 수 있습니다.

본 실험에서는 $\gamma(x)$에 대해 $L = 10$, $\gamma(d)$에 대해 $L = 4$ 로 설정하여 진행했습니다.

| 위치 벡터에 비교적 높은 L을 적용하는 이유
<br>장면의 기하학적 구조나 질감은 아주 미세한 위치 변화에도 급격하게 바뀔 수 있기 때문에 더 큰 주파수 사용. ex) 머리카락 한 올, 나뭇결 무늬

| 방향 벡터에 비교적 낮은 L을 적용하는 이유
<br>시점 의존적 효과 는 일반적으로 보는 각도에 따라 부드럽게 변하기 때문에 비교적 낮은 주파수 사용. ex) 하이라이트, 반사

### Hierarchical volume sampling

지금까지의 방식으로는 ray 당 $N$개의 질의 지점에서 뉴럴 래디언스 필드 네트워크를 빽빽하게 평가하는 전략은 비효율적입니다. 최종적으로 렌더링된 이미지에 기여하지 않는 빈 공간이나 가려진 영역도 반복적으로 샘플링 되기 때문입니다.

그래서 Coarse, Fine 두번으로 나누어 샘플링을 진행합니다. Coarse에서는 볼륨공간 상의 표면이 존재할 것 같은 위치를 제안하는 역할을 하며 Fine 에서 해당 제안을 바탕으로 효과적인 샘플링을 진행하게 됩니다.

**Coarse Network**

![Desktop View](/assets/img/nerf/coarse_network.webp)

$$\hat C_c(r)= \sum _{i=1}^{N_c}w_ic_i, w_i=T_i(1− \exp(− \sigma_i\delta_i))$$

Stratified samplling 으로 $N_c$개의 sample을 추출하고 가중치, $w_i$를 정규화합니다.
정규화된 가중치를통해 광선에 따라 구간별 확률 밀도함수를 얻을 수 있고, Fine Network에서 ray를 sampling 하는데에 사용합니다.


**Fine Network**

![Desktop View](/assets/img/nerf/fine_network.webp)

PDF를 통해 CDF 를 구하고 균일하게 점을 찍어 이에 상응하는 x를 찾아 샘플링하는 Inverse Transform Sampling 를 활용하여 ray를 sampling 합니다.

### Implementation details

$$L = \sum _{r\in R}[\left\| \hat C_c(r) - C(r) \right\|^2_2 + \left\| \hat C_f(r) - C(r) \right\|^2_2]$$

Coarse Network와 Fine Network 를 모두 평가한 다음, $N_c + N_f$개의 모든 sample을 사용하여 최종 렌더링 색상을 계산합니다.

최종 렌더링은 Fine Network에서 계산된 값을 통해 나오지만, coarse 네트워크의 가중치 분포가 Fine Network의 sample 할당에 사용될 수 있도록 Coarse Network의 손실도 최소화합니다.

**MLP Structure**

![Desktop View](/assets/img/nerf/mlp_structure.webp)

위 그림은 NeRF에서 사용된 MLP의 구조를 나타냅니다. 파란색 상자는 hidden layer를 표현하며, 초록색 상자는 input을, 빨간색 상자는 output을 표현합니다. $\gamma$는 positional encoding 함수를 나타냅니다. 또한 검정색 실선 화살표는 ReLU activation을, 검정색 점선 화살표는 sigmoid activation을, 주황색 화살표는 activation이 없음을 나타냅니다.

중요한 것은 3D point의 $x, d$가 MLP에 입력되는 타이밍과, 색상과 밀도의 출력 타이밍입니다. $x$만 갖고 layer를 투과하여 밀도가 예측되며, 밀도가 출력될 때 $d$ 정보까지 추가적으로 입력되어 색상 이 예측됩니다. 이는 NeRF의 밀도는 점의 위치에 따라 결정되고 색상은 점의 위치와 바라보는 방향에 따라 결정된다는 성질을 반영하고 있습니다.
또한 5번째 hidden layer에서 skip connection을 위해 한번 더 $x$를 대입합니다.

**NDC space dervation**

![Desktop View](/assets/img/nerf/ndc_dervation.webp){: width="350" }

만약 깊이를 기준으로 균일하게 sampling 한다면 먼 곳에서는 장면이 거의 변하지 않는데도 촘촘하게 sampling하여 계산자원을 낭비하게 됩니다.

이 문제를 해결하기 위해 NDC 공간으로 3D 공간을 의도적으로 왜곡하여 sampling에 반영합니다.
시차는 거리에 반비례하는 $1/z$ 로 표현 가능하며 NDC공간의 새로운 $z$축 좌표는 시차에 비례하게 됩니다.

## 6. Results
### Datasets
### Comparisons
### Discussion
### Ablation studies
## 7. Conclusion

NeRF는 복잡한 실사 데이터를 MLP상의 weight을 통해서 적은 크기로 표현할 수 있다는 점과, 깊이 및 3D 정보가 필요하지 않다는 점, 그리고 렌더링 과정이 미분 가능하기 때문에 gradient 기반 optimaization을 사용할 수 있다는 점으로 혁신적인 모델로 평가받고 있습니다.

하지만 학습시간이 매우 느리며, 정적인 scene에 대해서만 표현이 가능합니다. 또한 학습을 위해 사진들과 카메라 파라미터 정보가 필요하다는 단점이 존재합니다.

## Reference

> [!INFO] **참고할 만한 문서**
>
> [여기를 클릭하여 공식 문서를 확인하세요](https://example.com)
> 이 문서는 북마크 설정에 대한 상세 내용을 담고 있습니다.
