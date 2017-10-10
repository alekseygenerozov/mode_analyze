(* ::Package:: *)

BeginPackage["IApprox`"]
I0Approx::usage = "Approximations to integral Ilm appeaing in the tidal coupling constant--From Press&Teukolsky 1977"
Begin["`Private`"];
I0Approx[y_, 0, 0]:=\[Pi] (2^(1/2) y)^(-1/3 ) AiryAi[(2 ^(1/2) y)^(2/3)]

I0Approx[y_, 1, 0]:=Piecewise[{{ (1.5288+0.79192 y^(1/2)-0.86606 y+0.14593y^(3/2))/(1+1.6449 y^(1/2)-1.2345 y+0.19392 y^(3/2)) Exp[-2^(3/2)/3 y], y<=4}}, (1.4119+18.158 y^(1/2)+22.152 y)/(1+12.249 y^(1/2)+28.593 y) Exp[-2^(3/2)/3 y]]

I0Approx[y_, 2, 0]:=(0.78374 +1.5039 y^(1/2)+1.0073 y+0.71115 y^(3/2))/(1+1.9128 y^(1/2)+1.0384 y +1.2883 y^(3/2)) (1+2^(3/2)/3 y)^(1/2) Exp[-2^(3/2)/3 y]
I0Approx[y_, 3, 0]:=(0.58894+0.32381y^(1/2)+0.45605 y+0.1522 y^(3/2))/(1.+0.54766 y^(1/2)+0.7613y+0.53016 y^(3/2)) (1+2^(3/2)/3 y)Exp[-2^(3/2)/3 y]
I0Approx[y_, l_, 0]:=(2 l-3)/(2l -2) I0Approx[y, l-1,0]+y^2/((2 l -2)(l-3)) I0Approx[y, l-4,0]/;l>=4
I0Approx[y_, l_, m_]:=(2-2 (m-1)/l)I0Approx[y, l+1, m-1]-I0Approx[y, l, m-1]-2^(1/2) y/l I0Approx[y, l-1, m-1]/;m>0
I0Approx[y_, l_, m_]:=(2+2 (m+1)/l)I0Approx[y, l+1, m+1]-I0Approx[y, l, m+1]+2^(1/2) y/l I0Approx[y, l-1, m+1]/;m<0

I0[y_, l_, m_]:=NIntegrate[(1+x^2)^-l Cos[2^(1/2) y (x+x^3/3)+2m ArcTan[x]], {x, 0, \[Infinity]}]
SetAttributes[I0Approx, Listable];
End[];
EndPackage[];
