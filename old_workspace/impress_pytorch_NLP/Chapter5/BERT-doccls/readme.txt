---------------------------------------------------------
BERT ���g�����������ނ̃f�[�^
�ixtrain.pkl, ytrain.pkl, xtest.pkl, ytest.pkl�j
�̍���
---------------------------------------------------------
(1)

�ȉ��� URL ���� ldcc-20140209.tar.gz ���_�E�����[�h����B
https://www.rondhuit.com/download.html#ldcc

(2)

Lhaplus �Ȃǂ� ldcc-20140209.tar.gz ���𓀂���B
�ȉ��̃f�B���N�g���ƃt�@�C�����ł���B

    text/
      dokujo-tsushin/
      it-life-hack/
      kaden-channel/
      livedoor-homme/
      movie-enter/
      peachy/
      smax/
      sports-watch/
      topic-news/
      CHANGES.txt
      README.txt

(3)

text �̂���f�B���N�g���� mkldcc.py �����s����B
> python mkldcc.py
==>  train.tsv  �� test.tsv  ���쐬�����B

(4)

train.tsv  �� test.tsv ��  BERT-doccls �̃f�B���N�g�����ɃR�s�[����B

(5)

BERT-doccls �̃f�B���N�g�����ňȉ������s����B

> python mkdata11.py
==>  xtrain.pkl �� ytrain.pkl  ���쐬�����B

> python mkdata12.py
==>  xtest.pkl �� ytest.pkl  ���쐬�����B

