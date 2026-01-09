---
title: "[Review]Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing"
date: 2025-08-20 23:02:11 +0900
categories: [AI, 3D, Editting, Segmentation]
tags: [review]     # TAG names should always be lowercase
math: true
description: This is review for the Feature Splatting

---
> **ECCV 2024**
> 
> **Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing(1 April 2024)**
> 
> Ri-Zhao Qiu,  Ge Yang,   Weijia Zeng,  Xiaolong Wang  
> *UC San Diego \| MIT \| IAIFI*
>
> [![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)]( https://arxiv.org/abs/2404.01223 ) [![Page](https://img.shields.io/badge/Project-Page-blue)]( https://feature-splatting.github.io/) [![Code](https://img.shields.io/badge/Github-Code-181717?logo=github)]( https://github.com/vuer-ai/feature-splatting-inria ) 

 <br>
 
 
 이번 포스팅에서는 Feature Splatting에 대해서 리뷰해보고자 합니다. Feature Splatting에서는 View Synthesis를 자연어 기반으로 Editting 하기 위한 방법을 소개하고 있습니다. 

 
## Introduction

Feature Splatting은 다음에 대해서 핵심적인 기여를 했습니다.

- 외형, 기하, 의미 정보를 가우시안 표현으로 통합하여 최적화할 수 있습니다.
- 단순히 open-vocabulary segmentation을 넘어, 객체의 구성 요소에 대해, 학습된 특징을 기반으로 각 재료의 물리적 속성을 자동으로 결정할 수 있습니다.

다시말해 3D를 Segmentation 할 수 있고, 물리 기반으로 사실적인 움직임을 시뮬레이션할 수 있습니다.

## Related Work

1. Novel View Synthesis

- implicit methods : 대표적인 연구는 NeRF로, 장면의 radiance를 예측하기 위해 신경망을 학습하는 접근 방법이 있습니다.
- explicit methods : 대표적인 연구는 Gaussian Splatting으로 장면을 Gaussian의 집합으로 표현하는 방법이 있습니다.

2. Scene Editing with Distilled Feature Fields

- ClimateNeRF : 물리 시뮬레이션을 렌더링 과정에 삽입하여 다양한 기상 효과를 시뮬레이션하는 방법입니다.
- DFF : 제로샷 오픈 텍스트 분할을 통해 외관 편집을 수행하는 방법입니다.

## Language-Driven Physics-Based Synthesis and Editing

 ![Desktop View](/assets/img/feature-splatting/pipeline.png){: .normal}

Feature Splatting에서 “Language-Driven Physics-Based Scene Synthesis and Editing” 을 수행하기 위해서 아래의 세가지 단계를 설명하고 있습니다.

1. **Feature Splatting** : VLM에서 추출한 semantic feature를 Gaussian으로 distill 하는 방법.
2. **Language-grounded Decomposition** : open-text query를 통해서 객체를 구성요소 단위로 분리하는 방법.
3. **Physics Sim** : 언어 기반으로 물질의 재료에 대한 속성을 자동으로 부여하고 물리 엔진 기반으로 시뮬레이션하는 방법.

즉, Gaussian을 학습하는 과정에서 VLM을 가지고 각 Gaussian에 의미정보를 추가하여 학습함으로써 segmentation을 수행할 수 있도록 하고, 선택한 객체뿐만 아니라 객체의 구성요소 단위로 분해하여 각각의 재료에 해당하는 속성을 부여하는 작업을 수행하게 됩니다. 이제 이 객체를 재료점법(Material Point Method)을 통해서 물리 기반의 사실적인 움직임을 시뮬레이션 할 수 있습니다.

다음 내용에서 각 단계별로 자세하게 알아보도록 하겠습니다.

##  Differentiable Feature Splatting

### Feature Splatting

논문의 제목과도 같은 Feature Splatting 단계에서는 각각의 Gaussian 들에게 의미 정보를 부여함으로써 언어 기반으로 Segmentation을 수행하기위한 단계 입니다. 이를 위해서 사전 학습된 대규모 Vision — Language model(VLM)을 가지고 학습을 진행하게 됩니다.

$$ \{\hat{\mathbf F}, \hat{\mathbf C}\} = \sum_{i \in N} \{ \mathbf f_i, \mathbf c_i \} \cdot \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j) $$

이 과정은 Gaussian Splatting의 Rasterization 단계에서 진행됩니다. 아래의 식에서 볼 수 있듯이 이전과 다른 점은 색상 (C)뿐만 아니라 Semantic Feature 벡터(F) 또한 추가로 학습되어진다는 점 입니다.

이해가 더 잘 될 수 있도록 Rasterization 과정을 포함하여 정리해보겠습니다. Rasterization 과정의 목표는 입력 이미지와 픽셀단위로 비교를 진행하고 그 오차를 가지고 학습을 진행할 수 있도록 하기 위함입니다. 따라서 입력 이미지에 해당하는 카메라 시점의 렌더링된 뷰에 대해서 다음의 연산을 진행하게 됩니다.

1. View Frustum 밖에 벗어난 Gaussian들을 제거하고 , 16 x 16 의 크기로 타일을 나눕니다.
2. 각 타일별로 Gaussian 들을 정렬하고, 영상 평면으로 투영하여 입력 이미지와 동일하게 만들고 싶은 뷰가 그려지게 됩니다.
3. 이 때, 투영과정에서 겹쳐진 각 가우시안들은 α-blending 공식을 통해서 색상이 결정되게 되는데 한 픽셀에서 겹쳐진 가우시안의 총 갯수는 N, 정렬된 각 가우시안의 순서번호는 i, 카메라와 가장 가까운 가우시안은 j 로써 계산되어집니다.
4. 여기까지가 Gaussian Splatting에서 설명하는 Rasterization의 과정이었고, Feature Splatting에서는 색상뿐만 아니라 의미 특징 벡터 f 와 함께 계산된다는 차이점이 존재합니다.

```python

def __init__(self, sh_degree : int, distill_feature_dim : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.distill_feature_dim = distill_feature_dim
        self._xyz = torch.empty(0)  # (N, 3)
        # self.get_features contains both features_dc and features_rest
        self._features_dc = torch.empty(0)  # (N, 8, 3)
        self._features_rest = torch.empty(0)  # (N, 8 3)
        self._scaling = torch.empty(0)  # (N, 3); pass through sigmoid before return
        self._rotation = torch.empty(0)  # (N, 4); quaternion; pass through L2 normalization before return
        self._opacity = torch.empty(0)  # (N, 1)
        self._distill_features = torch.empty(0)  # (N, distill_feature_dim)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
```
위 코드는 feature-splatting-inria/scene/gaussian_model.py 의 일부입니다. gaussian_model.py 는 각 가우시안을 모델링하는 부분이며, self._distill_features 가 바로 각 가우시안이 갖게되는 Semantic Feature 에 해당합니다.

렌더링된 이미지에서 색상 값은 입력 이미지를 통해서 오차를 구할 수 있지만, feature 벡터는 그럴 수 없습니다. 따라서 사전학습된 VLM 모델을 통해 정답지를 계산해 두어야 합니다.

### Improving Reference Feature Quality Using Part-Priors

 ![Desktop View](/assets/img/feature-splatting/training.webp)
 
 Feature Splatting 에서는 다음과 같은 방법으로 feature에 대한 오차를 구합니다.

1. Rasterization된 특정 픽셀에 대해서 Semantc feature(z)가 두개의 브랜치로 나뉩니다.
2. 입력 이미지로부터 얻은 (1)SAM으로 보강한 CLIP feature map 과 (2)DINO를 통해 얻은 feature map 으로부터 각 브랜치와의 Cosine Loss 를 구합니다. 즉, Feature Splatting에서는 (1)과 (2) 두개의 정답지를 계산합니다.
3. 구해진 두 Loss 값을 적절히 보강하여 하나의 Loss로 반영되어 학습되어집니다.

더 자세한 방법을 알아보기 전에 CLIP, DINO, SAM에 대한 모델이 어떤 역할을 하는지 아주 간략하게 살펴보겠습니다.

CLIP vs DINO vs SAM

- **CLIP**은 이미지와 텍스트를 같은 의미 공간에서 이해하는 모델입니다. 즉, ‘강아지’ 사진과 “강아지”라는 텍스트를 서로 연결할 수 있어, 텍스트만으로 이미지를 검색할 수 있습니다. 사전에 대규모 이미지 — 텍스트 쌍의 데이터 셋을 가지고 학습이 되었습니다.
- **DINO**는 정답 없이 이미지만을 보고 스스로 학습하는 자기 지도 학습 모델입니다. 따라서 정답이 무엇인지는 알 수 없으나 의미 정보를 가지고 분류할 수 있습니다.
- **SAM**은 Segmentation을 위한 모델입니다. 이미지로부터 객체의 외곽선을 따내는 마스크를 출력합니다.

**1. SAM으로 보강한 CLIP feature map**

 ![Desktop View](/assets/img/feature-splatting/feature_model.webp)
 
 단순히 CLIP만 사용하지 않고 SAM을 추가적으로 사용한 것은 깔끔하게 segmentation 하기 위함입니다. CLIP 모델만 사용했을 경우 위 사진의 (b)와 같이 명확한 경계선을 찾는 것에는 무리가 있습니다. 따라서 둘을 같이 사용한다면 (d)처럼 의미정보와 함께 명확한 경계선을 포함한 segmentation을 할 수 있습니다. 구체적인 방법은 다음과 같습니다.

SAM을 통해 입력 이미지의 마스크 집합 {M}을 추출합니다. 마스크 M은 이진마스크로 경계선 안에서 1, 밖에서 0의 값을 가지므로 CLIP을 통해 추출한 Feature 벡터와 결합한다면 경계선 안에서 의미정보를 가질 수 있도록 만들 수 있습니다.

$$ w = \text{MAP}(M, F_C) = \frac{\sum_{i \in F_C} M(i)\cdot \frac{F_C(i)}{\|F_C(i)\|}}{\sum_{i \in F_C} M(i)} $$

위 식에서 $w$는 Segmentation 내에서 갖는 Feature 벡터를 의미하며 마스크 $M$과 Feature벡터 $F_c$ 에의해 결정됩니다. MAP(Masked Average Pooling)에서 $i$는 입력 이미지 내의 모든 픽셀을 의미하며 동일한 $M$ 안에 위치해 있다면 동일한 값 $w$을 갖습니다. Feature벡터를 정규화한 이유는 $w$를 구하기 위해 다른 Feature 벡터들과 연산(평균)을 취하기 위함입니다.

결국 계산된 $w$는 $M$에 의해 세그멘테이션된 내부의 모든 픽셀에 할당됩니다.이 때, 한 픽셀이 여러 의미를 가지면 관련된 모든 $w$의 평균을 취합니다. 이렇게 하면 Feature 벡터 학습을 위한 첫번째 정답지를 얻을 수 있습니다.

**2. DINO를 통해 얻은 feature map**

두번째 정답지는 DINO로 부터 얻은 Feature Map입니다. 두번째 정답지가 있는 이유는 **SAM으로 보강한 CLIP feature map** 에 과적합하지 않도록 하기 위함이며, 얕은 MLP 를 도입하여 중간 렌더링 단계에서 얻은 Feature를 입력으로 두개의 출력 브랜치를 냅니다. (그림에서는 1x1 convolution network를 사용했다고 명시되어 있습니다.)

$$ \hat{\mathbf F}_C, \hat{\mathbf F}_D = \text{MLP}(\hat{\mathbf F}) $$

위의 식에서 $F_C$는 CLIP에 대한 Feature 벡터이고 $F_D$는 DINO에 대한 Feature 벡터입니다. 손실 함수는 두개 모두 Cosine Loss로 학습되고, 언어기반에 집중하기 위해 DINO 손실 오차의 가중치를 줄여서 사용합니다.

$$ \mathcal L_\text{CLIP} + \lambda \cdot \mathcal L_\text{DINO} $$ 

## Language-guided Scene Decomposition

이제 학습된 각 가우시안들을 질의한 단어에 대해서 선택하고, 해당 단어가 의미하는 객체의 구성요소 단위로 분해하는 과정에 대해서 알아보겠습니다. 이 과정은 양성 질의(positive query)와 음성 질의(negative query)를 통해서 수행하게 됩니다.

예를 들어 꽃이 꽂힌 꽃병을 선택하고 싶을 때는 양성질의로 “a vase with flowers”, 음성질의로 “object”와 같이 설정하여 일반적인 것들과 구분하여 선택을 명확하게 합니다.

꽃을 분리하여 선택하고자 할 때에는 양성질의로 “flower”, 음성 질의로 “vase” 로 하여 꽃병을 배제하여 꽃만 분리할 수 있도록 합니다. 구체적으로 이 과정은 다음 단계를 통해서 수행됩니다.

![img-descrption](/assets/img/feature-splatting/temperature.webp)
_렌더링된 이미지(좌) 와 temperatured softmax로 표현된 이미지(우)_

1. 입력받은 단어를 통해 CLIP의 텍스트 임베딩을 얻고, 각 가우시안의 렌더링된 CLIP 특징과 텍스트 임베딩 간의 Cosine 유사도를 계산합니다.
2. 유사도 값을 통해 temperatured softmax 계산하여 확률 분포를 얻습니다.
3. 임계값을 초과하는 가우시안을 선택하게 됩니다.

 ![Desktop View](/assets/img/feature-splatting/editting.webp)

이렇게 분해한 객체들을 편집하는 방법은 각 가우시안의 속성을 조정함으로써 이룰 수 있습니다. 여기에서는 객체 제거, 이동, 회전, 스케일링 을 다루고 있습니다.

편집 대상으로 선택된 가우시안의 집합을 위와 같이 표현할 수 있을 때, 객체 편집은 다음과 같이 수행됩니다.

$$ \{\hat X, \hat \Sigma \} \subseteq \{X, \Sigma\} $$ 

- **객체 제거 (Object Removal)**: 객체 제거는 선택된 가우시안들을 단순히 삭제하는 과정을 통해 수행합니다.
    
    $$
    \{X, \Sigma\} := \{X, \Sigma\} \setminus \{\hat X, \hat \Sigma\}
    $$
    
- **이동 (Translation)**: 주어진 변위 벡터 $b_1 \in \mathbb{R}^3$에 따라, 가우시안의 중심을 이동시킵니다.
    
    $\hat X := \hat X + b_1$
    
- **회전 (Rotation)**: 주어진 회전 행렬 $R_1 \in SO(3)$에 따라, 선택된 가우시안의 공분산을 수정합니다.
    
    $\hat \Sigma := R_1 \hat R \hat S \hat S^\top \hat R^\top R_1^\top$
    
- **스케일링 (Scaling)**: 주어진 축 정렬 스케일 벡터 $s_1 \in \mathbb{R}^3$에 따라, 객체의 크기를 조정합니다.
    
    $\hat X = s_1 \hat X,\; \hat \Sigma := \hat R (s_1 \hat S)(s_1 \hat S)^\top \hat R^\top$

## Language-Driven Physics Synthesis

 ![Desktop View](/assets/img/feature-splatting/physics_piepline.webp)

이제 우리는 언어기반으로 객체를 분해하고 편집할 수 있습니다. 이제 어떻게 분해한 객체에 대해서 물리기반의 사실적인 시뮬레이션이 가능한지 알아보겠습니다. 다음은 언어기반 물리 시뮬레이션의 파이프라인입니다.

1. 시뮬레이션을 위한 **객체 분리** 단계를 수행합니다.
2. **물리기반 전처리**를 진행합니다.
3. **재료점법 기반의 물리 엔진으로 시뮬레이션**을 가능하게 합니다.

### 시뮬레이션을 위한 객체 분리

이 단계에서는 목재, 세라믹, 강철 등과 같은 일반적인 강체를 위한 재료 단어 집합을 구성합니다. 선택된 다중 파트 객체(예 : “a vase with flowers”)가 주어지면 추가적인 CLIP 유사도 비교를 통해서 객체의 일부중 더 밀접한 부분을 선택합니다.

 ![img-description](/assets/img/feature-splatting/object_decomposition.webp)
 _바닥의 충돌 직전의 공(초록) 과 충돌 후 무너진 공(빨강)_

구체적인 단계에 대해서 설명하기 이전에 3D Gaussian이 갖고 있는 특성 및 한계점에 대해서 알고 있어야 무엇을 해결하고자 전처리 과정을 수행했는지 이해할 수 있습니다. 3D Gaussian으로 만들어진 객체는 기본적으로 속이 텅 빈 껍데기와 같습니다. 따라서, 이 Gaussian들을 단순한 점 입자로 취급해 물리 시뮬레이션을 적용하면 두 가지 문제가 발생합니다.

- 내부 지지가 부족하여, 객체가 충돌 표면에 닿을 때 무너짐 현상
- 객체가 변형될 때 원치 않는 아티팩트 발생.

### 물리기반 전처리 — 언어 기반 충돌 표면 추정

충돌 표면을 추정하기 위해 “바닥” 이나 “테이블” 같은 일반적인 평면 객체를 장면 분해 파이프라인에 넣어서 법선 벡터를 구하고 중력 방향으로 설정합니다.

### 물리기반 전처리 — 암묵적 부피 보존

Gaussian으로 구성한 3D는 내부가 비어있기 때문에 발생하는 문제점이 있다고 언급한 바 있습니다. 그래서 다음과 같이 암묵적으로 부피 보존을 수행하여 이를 해결합니다.

1. 가우시안의 모양과 불투명도 정보를 이용해 객체의 표면을 더 촘촘하게 만듭니다.
2. 이 촘촘해진 표면을 기준으로, 객체 중심부터 표면까지 투명한 내부 입자들을 가득 채워 넣습니다.

투명한 입자를 사용하기 때문에 렌더링 과정에서 영향을 미치지 않아 기존과 동일한 렌더링 성능을 유지하게 합니다.

### 물리기반 전처리 — 회전 추정

정확한 회전을 추정하기 위해 한 점이 아닌 주변 이웃 가우시안들과의 관계를 이용합니다.

1. 시뮬레이션할 각 가우시안에 대해 가장 가까운 두 이웃을 찾는다.
2. 이 세 개 가우시안의 중심이 이루는 평면을 정의한다.
3. 그 객체가 움직일 때 법선벡터가 어떻게 변하는지 추적하여, 가우시안의 회전 값으로 사용합니다.

이렇게 전처리된 3D Gaussian 들을 Gaussian에 적응된 MPM(Material Point Method) 엔진을 사용하여 재료에 기반한 물리 시뮬레이션을 가능하게 할 수 있습니다.

 ![Desktop View](/assets/img/feature-splatting/mpm.webp)

MPM 엔진은 입자와 격자가 가진 장점을 결합해 만든 물리 엔진 입니다. 입자는 각 물질의 고유한 속성(질량, 속도, 색상 등)을 저장하고 격자위에서 충돌과 같은 힘과 관련된 계산을 진행하게 됩니다. 즉, 입자가 가진 속성 정보를 격자에서 계산하고, 계산된 결과를 입자에 전파하는 방식으로 물리 시뮬레이션을 수행할 수 있습니다. 여기서는 MPM에 대한 설명은 간단하게 다루고 넘어가겠습니다.

## Experiments

물리 편집의 경우, 비교할 수 있는 정답 이미지나 기존 벤치마크가 없으므로 정성적 비교에 초점을 두었다고 하며, 정량적 평가로는 Segmentation한 객체의 위치 정확도, 3D 품질 평가 등에 사용했습니다.

 ![img-description](/assets/img/feature-splatting/position_correctness.webp)
_Segmentation한 객체의 위치 정확도_

위치 정확도는 LERF 에서 사용한 위치 정확도 측정 프로토콜을 사용하였으며, 렌더링된 2D 에서의 정확도는 약 81.7%로 기존의 다른 모델들과 비교했을 때 높은 수준임을 증명했습니다.

 ![img-description](/assets/img/feature-splatting/quality_rendering.webp)
_3D 품질 평가 및 렌더링 속도_

PSNR은 26.47로 이전의 모델과 비슷한 수준으로 유지하였으며, 렌더링 속도 또한 102fps 로 유지하였습니다. 이는 학습 과정에 Semantic Feature를 추가한 점, 암묵적 부피 보존 과정에서 투명한 입자들을 채워넣은 점들을 반영했을 때 의미 있는 결과라고 생각합니다.

물리 편집 측면은 다음과 같이 정성적으로 평가하였습니다.

**지능적 물리 효과**

같은 객체 내에서도 ‘꽃병’은 단단한 물체로, ‘꽃’은 바람에 흔들리는 부드러운 물체로 자동으로 구분하여 움직임을 부여합니다.

**사실적인 움직임**

객체 내부를 가상 입자로 채워 부피를 보존함으로써, 공이 땅에 튕길 때 찌그러지지 않고 사실적으로 움직일 수 있습니다.

**실시간 성능**

물리 계산 시에는 약 30 fps, 이미 계산된 움직임을 재생할 때는 약 100 fps의 빠른 속도로 실시간 렌더링이 가능합니다.

**기하학 편집**

텍스트로 객체를 선택한 후, 제거, 크기 조절, 회전, 이동, 복제 등의 작업을 자동으로 수행합니다.

**외관 편집**

배경은 그대로 두고 특정 객체의 색상이나 스타일만 빠르고 정확하게 변경 가능합니다.

## Conclusion

Feature Splatting은 객체 제거 또는 이동 후 배경에 발생할 수 있는 아티팩트가 존재한다는 한계점이 있지만, 물리 시뮬레이션, 기하학 및 외관 편집에 대한 효과성을 입증 할 수 있었습니다.

결론적으로, Feature Splatting과 같은 기술이 건축 공정 관리에 가져올 미래는 단순히 3D 모델을 ‘보는’ 것을 넘어, 우리가 만든 가상현실과 ‘대화’하는 시대로의 전환을 의미합니다. 현장의 상황을 스스로 인지하고, 우리의 질문에 답하며, 잠재적인 문제점을 미리 알려주는 파트너가 될 수 있기를 꿈꾸며 이 글을 마칩니다.

## Reference

{% linkpreview "https://xoft.tistory.com/99" %}

