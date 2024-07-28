// Made with Amplify Shader Editor v1.9.0.2
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/PT_Vegetation_Flowers_Shader"
{
	Properties
	{
		[NoScaleOffset]_BASETEXTURE("BASE TEXTURE", 2D) = "black" {}
		[Toggle]_CUSTOMCOLORSTINTING("CUSTOM COLORS  TINTING", Float) = 1
		_TopColor("Top Color", Color) = (0.3505436,0.5754717,0.3338822,1)
		_GroundColor("Ground Color", Color) = (0.1879673,0.3113208,0.1776878,1)
		[HDR]_Gradient(" Gradient", Range( 0 , 1)) = 1
		_GradientPower1("Gradient Power", Range( 0 , 10)) = 1
		_LeavesThickness("Leaves Thickness", Range( 0.1 , 0.95)) = 0.5
		[Toggle]_CUSTOMFLOWERSCOLOR("CUSTOM FLOWERS COLOR", Float) = 0
		[HideInInspector]_MaskClipValue("Mask Clip Value", Range( 0 , 1)) = 0.5
		[HDR]_FLOWERSCOLOR("FLOWERS COLOR", Color) = (1,0,0,0)
		[Toggle(_TRANSLUCENCYONOFF_ON)] _TRANSLUCENCYONOFF("TRANSLUCENCY ON/OFF", Float) = 1
		[Header(Translucency)]
		_Translucency("Strength", Range( 0 , 50)) = 1
		_TransNormalDistortion("Normal Distortion", Range( 0 , 1)) = 0.1
		_TransScattering("Scaterring Falloff", Range( 1 , 50)) = 2
		_TransDirect("Direct", Range( 0 , 1)) = 1
		_TransAmbient("Ambient", Range( 0 , 1)) = 0.2
		_TransShadow("Shadow", Range( 0 , 1)) = 0.9
		[Toggle(_CUSTOMWIND_ON)] _CUSTOMWIND("CUSTOM WIND", Float) = 1
		_WindMovement("Wind Movement", Range( 0 , 1)) = 0.5
		_WindDensity("Wind Density", Range( 0 , 5)) = 0.2
		_WindStrength("Wind Strength", Range( 0 , 1)) = 0.3
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "TransparentCutout"  "Queue" = "Geometry+0" }
		Cull Off
		CGPROGRAM
		#include "UnityShaderVariables.cginc"
		#include "UnityPBSLighting.cginc"
		#pragma target 3.0
		#pragma multi_compile_instancing
		#pragma shader_feature _CUSTOMWIND_ON
		#pragma shader_feature_local _TRANSLUCENCYONOFF_ON
		#pragma multi_compile __ LOD_FADE_CROSSFADE
		#pragma surface surf StandardCustom keepalpha addshadow fullforwardshadows exclude_path:deferred dithercrossfade vertex:vertexDataFunc 
		struct Input
		{
			float2 uv_texcoord;
			float3 worldPos;
		};

		struct SurfaceOutputStandardCustom
		{
			half3 Albedo;
			half3 Normal;
			half3 Emission;
			half Metallic;
			half Smoothness;
			half Occlusion;
			half Alpha;
			half3 Translucency;
		};

		uniform float _WindMovement;
		uniform float _WindDensity;
		uniform float _WindStrength;
		uniform float _CUSTOMCOLORSTINTING;
		uniform float _CUSTOMFLOWERSCOLOR;
		uniform sampler2D _BASETEXTURE;
		uniform float4 _FLOWERSCOLOR;
		uniform float4 _GroundColor;
		uniform float4 _TopColor;
		uniform float _Gradient;
		uniform float _GradientPower1;
		uniform half _Translucency;
		uniform half _TransNormalDistortion;
		uniform half _TransScattering;
		uniform half _TransDirect;
		uniform half _TransAmbient;
		uniform half _TransShadow;
		uniform float _LeavesThickness;
		uniform float _MaskClipValue;


		float3 mod2D289( float3 x ) { return x - floor( x * ( 1.0 / 289.0 ) ) * 289.0; }

		float2 mod2D289( float2 x ) { return x - floor( x * ( 1.0 / 289.0 ) ) * 289.0; }

		float3 permute( float3 x ) { return mod2D289( ( ( x * 34.0 ) + 1.0 ) * x ); }

		float snoise( float2 v )
		{
			const float4 C = float4( 0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439 );
			float2 i = floor( v + dot( v, C.yy ) );
			float2 x0 = v - i + dot( i, C.xx );
			float2 i1;
			i1 = ( x0.x > x0.y ) ? float2( 1.0, 0.0 ) : float2( 0.0, 1.0 );
			float4 x12 = x0.xyxy + C.xxzz;
			x12.xy -= i1;
			i = mod2D289( i );
			float3 p = permute( permute( i.y + float3( 0.0, i1.y, 1.0 ) ) + i.x + float3( 0.0, i1.x, 1.0 ) );
			float3 m = max( 0.5 - float3( dot( x0, x0 ), dot( x12.xy, x12.xy ), dot( x12.zw, x12.zw ) ), 0.0 );
			m = m * m;
			m = m * m;
			float3 x = 2.0 * frac( p * C.www ) - 1.0;
			float3 h = abs( x ) - 0.5;
			float3 ox = floor( x + 0.5 );
			float3 a0 = x - ox;
			m *= 1.79284291400159 - 0.85373472095314 * ( a0 * a0 + h * h );
			float3 g;
			g.x = a0.x * x0.x + h.x * x0.y;
			g.yz = a0.yz * x12.xz + h.yz * x12.yw;
			return 130.0 * dot( m, g );
		}


		void vertexDataFunc( inout appdata_full v, out Input o )
		{
			UNITY_INITIALIZE_OUTPUT( Input, o );
			float4 ase_vertex4Pos = v.vertex;
			float simplePerlin2D321 = snoise( (ase_vertex4Pos*1.0 + ( _Time.y * _WindMovement )).xy*_WindDensity );
			simplePerlin2D321 = simplePerlin2D321*0.5 + 0.5;
			float4 appendResult329 = (float4(( ( ( ( simplePerlin2D321 - 0.5 ) / 10.0 ) * _WindStrength ) + ase_vertex4Pos.x ) , ase_vertex4Pos.y , ase_vertex4Pos.z , 1.0));
			float4 lerpResult330 = lerp( ase_vertex4Pos , appendResult329 , ( ase_vertex4Pos.y * 2.0 ));
			float4 transform331 = mul(unity_WorldToObject,float4( _WorldSpaceCameraPos , 0.0 ));
			float4 temp_cast_2 = (transform331.w).xxxx;
			#ifdef _CUSTOMWIND_ON
				float4 staticSwitch333 = ( lerpResult330 - temp_cast_2 );
			#else
				float4 staticSwitch333 = ase_vertex4Pos;
			#endif
			float4 WIND393 = staticSwitch333;
			v.vertex.xyz = WIND393.xyz;
			v.vertex.w = 1;
		}

		inline half4 LightingStandardCustom(SurfaceOutputStandardCustom s, half3 viewDir, UnityGI gi )
		{
			#if !defined(DIRECTIONAL)
			float3 lightAtten = gi.light.color;
			#else
			float3 lightAtten = lerp( _LightColor0.rgb, gi.light.color, _TransShadow );
			#endif
			half3 lightDir = gi.light.dir + s.Normal * _TransNormalDistortion;
			half transVdotL = pow( saturate( dot( viewDir, -lightDir ) ), _TransScattering );
			half3 translucency = lightAtten * (transVdotL * _TransDirect + gi.indirect.diffuse * _TransAmbient) * s.Translucency;
			half4 c = half4( s.Albedo * translucency * _Translucency, 0 );

			SurfaceOutputStandard r;
			r.Albedo = s.Albedo;
			r.Normal = s.Normal;
			r.Emission = s.Emission;
			r.Metallic = s.Metallic;
			r.Smoothness = s.Smoothness;
			r.Occlusion = s.Occlusion;
			r.Alpha = s.Alpha;
			return LightingStandard (r, viewDir, gi) + c;
		}

		inline void LightingStandardCustom_GI(SurfaceOutputStandardCustom s, UnityGIInput data, inout UnityGI gi )
		{
			#if defined(UNITY_PASS_DEFERRED) && UNITY_ENABLE_REFLECTION_BUFFERS
				gi = UnityGlobalIllumination(data, s.Occlusion, s.Normal);
			#else
				UNITY_GLOSSY_ENV_FROM_SURFACE( g, s, data );
				gi = UnityGlobalIllumination( data, s.Occlusion, s.Normal, g );
			#endif
		}

		void surf( Input i , inout SurfaceOutputStandardCustom o )
		{
			float2 uv_BASETEXTURE2 = i.uv_texcoord;
			float4 tex2DNode2 = tex2D( _BASETEXTURE, uv_BASETEXTURE2 );
			float grayscale313 = dot(tex2DNode2.rgb, float3(0.299,0.587,0.114));
			float2 temp_cast_1 = (0.5).xx;
			float2 uv_TexCoord204 = i.uv_texcoord + temp_cast_1;
			float2 temp_cast_2 = (0.0).xx;
			float2 uv_TexCoord235 = i.uv_texcoord + temp_cast_2;
			float temp_output_238_0 = ( step( uv_TexCoord204.x , 1.0 ) + step( uv_TexCoord235.y , 0.5 ) );
			float clampResult248 = clamp( ( 1.0 - temp_output_238_0 ) , 0.0 , 1.0 );
			float FLOWERMASK395 = clampResult248;
			float4 temp_cast_3 = (( grayscale313 * FLOWERMASK395 )).xxxx;
			float4 blendOpSrc310 = _FLOWERSCOLOR;
			float4 blendOpDest310 = temp_cast_3;
			float4 lerpBlendMode310 = lerp(blendOpDest310,( blendOpSrc310 * blendOpDest310 ),FLOWERMASK395);
			float4 lerpResult341 = lerp( tex2DNode2 , ( saturate( lerpBlendMode310 )) , FLOWERMASK395);
			float3 ase_vertex3Pos = mul( unity_WorldToObject, float4( i.worldPos , 1 ) );
			float clampResult354 = clamp( pow( ( (0.5 + (ase_vertex3Pos.y - 0.0) * (2.0 - 0.5) / (1.0 - 0.0)) * _Gradient ) , _GradientPower1 ) , 0.0 , 1.0 );
			float4 lerpResult356 = lerp( _GroundColor , _TopColor , clampResult354);
			float4 GRADIENT380 = lerpResult356;
			float4 temp_cast_4 = (temp_output_238_0).xxxx;
			float4 lerpResult205 = lerp( GRADIENT380 , temp_cast_4 , FLOWERMASK395);
			float4 GRADIENTMASK402 = lerpResult205;
			float4 temp_cast_5 = (grayscale313).xxxx;
			float4 lerpResult362 = lerp( temp_cast_5 , (( _CUSTOMFLOWERSCOLOR )?( lerpResult341 ):( tex2DNode2 )) , FLOWERMASK395);
			float4 blendOpSrc232 = GRADIENTMASK402;
			float4 blendOpDest232 = lerpResult362;
			float clampResult281 = clamp( temp_output_238_0 , 0.0 , 1.0 );
			float FLOWERMASKINVERT398 = clampResult281;
			float4 lerpBlendMode232 = lerp(blendOpDest232,(( blendOpDest232 > 0.5 ) ? ( 1.0 - 2.0 * ( 1.0 - blendOpDest232 ) * ( 1.0 - blendOpSrc232 ) ) : ( 2.0 * blendOpDest232 * blendOpSrc232 ) ),FLOWERMASKINVERT398);
			float4 FINALCOLOR400 = (( _CUSTOMCOLORSTINTING )?( lerpBlendMode232 ):( (( _CUSTOMFLOWERSCOLOR )?( lerpResult341 ):( tex2DNode2 )) ));
			o.Albedo = FINALCOLOR400.rgb;
			float temp_output_407_0 = 0.0;
			o.Metallic = temp_output_407_0;
			o.Smoothness = temp_output_407_0;
			#ifdef _TRANSLUCENCYONOFF_ON
				float4 staticSwitch390 = ( FINALCOLOR400 * 1.0 );
			#else
				float4 staticSwitch390 = float4( 0,0,0,0 );
			#endif
			float4 TRANSLUCENCY391 = staticSwitch390;
			o.Translucency = TRANSLUCENCY391.rgb;
			o.Alpha = 1;
			float ALPHA382 = tex2DNode2.a;
			float TRANSPARENCY384 = ( 1.0 - step( ALPHA382 , ( 1.0 - _LeavesThickness ) ) );
			clip( TRANSPARENCY384 - _MaskClipValue );
		}

		ENDCG
	}
	Fallback "Diffuse"
}
/*ASEBEGIN
Version=19002
190;323;1496;706;574.5507;116.8802;1.3;True;False
Node;AmplifyShaderEditor.CommentaryNode;266;-2509,-592.5564;Inherit;False;1868.886;616.2441;mask;15;398;281;395;248;381;205;209;238;236;207;234;208;204;235;233;;1,1,1,1;0;0
Node;AmplifyShaderEditor.RangedFloatNode;206;-2517.34,-532.2863;Inherit;False;Constant;_Float0;Float 0;11;0;Create;True;0;0;0;False;0;False;0.5;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;233;-2487.624,-204.2801;Inherit;False;Constant;_Float2;Float 2;11;0;Create;True;0;0;0;False;0;False;0;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;235;-2192.235,-257.7668;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TextureCoordinatesNode;204;-2195.312,-547.0745;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;208;-2219.372,-332.4665;Float;False;Constant;_Float1;Float 1;11;0;Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;234;-2186.464,-79.52873;Inherit;False;Constant;_Float3;Float 3;11;0;Create;True;0;0;0;False;0;False;0.5;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;342;-1898.119,-1591.207;Inherit;False;1233.876;845.2716;GRADIENT;11;380;356;354;348;352;345;355;344;343;406;405;;1,1,1,1;0;0
Node;AmplifyShaderEditor.PosVertexDataNode;343;-1876.877,-1133.999;Inherit;False;0;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.StepOpNode;236;-1889.913,-286.8752;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.StepOpNode;207;-1901.069,-550.4159;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;238;-1759.072,-540.0843;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;344;-1876.52,-921.5023;Float;False;Property;_Gradient; Gradient;4;1;[HDR];Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;348;-1647.678,-1120.507;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0.5;False;4;FLOAT;2;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;405;-1614.903,-829.9576;Inherit;False;Property;_GradientPower1;Gradient Power;5;0;Create;True;0;0;0;False;0;False;1;10;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;209;-1706.885,-326.0652;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;39;-3987.263,88.08234;Inherit;False;3463.052;395.2417;COLOR;17;399;397;400;357;362;335;396;336;310;341;308;313;2;127;403;232;382;;1,1,1,1;0;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;345;-1433.938,-1009;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.PowerNode;406;-1286.285,-1005.241;Inherit;False;False;2;0;FLOAT;0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ClampOpNode;248;-1533.901,-330.3499;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;127;-3899.18,194.1785;Inherit;True;Property;_BASETEXTURE;BASE TEXTURE;0;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;black;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.RegisterLocalVarNode;395;-1256.425,-315.8854;Inherit;False;FLOWERMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ClampOpNode;354;-1132.375,-1026.321;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;314;-4261.057,888.3673;Inherit;False;3755.488;634.5623;WIND;21;333;332;331;330;329;328;327;326;325;324;323;322;321;320;319;318;317;316;315;393;404;;1,1,1,1;0;0
Node;AmplifyShaderEditor.ColorNode;352;-1813.582,-1534.757;Float;False;Property;_GroundColor;Ground Color;3;0;Create;True;0;0;0;False;0;False;0.1879673,0.3113208,0.1776878,1;0.05298166,0.3490566,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;355;-1807.839,-1340.912;Float;False;Property;_TopColor;Top Color;2;0;Create;True;0;0;0;False;0;False;0.3505436,0.5754717,0.3338822,1;0.01743852,0.5754717,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SamplerNode;2;-3448.14,212.932;Inherit;True;Property;_TextureSample0;Texture Sample 0;1;0;Create;True;0;0;0;False;0;False;-1;None;None;True;0;False;black;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;315;-4197.537,1357.597;Inherit;False;Property;_WindMovement;Wind Movement;19;0;Create;True;0;0;0;False;0;False;0.5;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleTimeNode;316;-4188.459,1274.563;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;396;-2510.315,403.643;Inherit;False;395;FLOWERMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCGrayscale;313;-2885.698,125.0036;Inherit;True;1;1;0;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;356;-1160.424,-1424.227;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;317;-3971.471,1220.502;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;318;-3942.178,965.9261;Inherit;False;1;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;308;-2691.317,263.3746;Inherit;False;Property;_FLOWERSCOLOR;FLOWERS COLOR;9;1;[HDR];Create;True;0;0;0;False;0;False;1,0,0,0;1,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;336;-2498.955,127.7875;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;380;-856.7367,-1437.412;Inherit;False;GRADIENT;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;381;-1220.036,-579.8229;Inherit;False;380;GRADIENT;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;320;-3846.511,1353.231;Inherit;False;Property;_WindDensity;Wind Density;20;0;Create;True;0;0;0;False;0;False;0.2;1.91;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.ScaleAndOffsetNode;319;-3814.034,1112.694;Inherit;True;3;0;FLOAT4;0,0,0,0;False;1;FLOAT;1;False;2;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.BlendOpsNode;310;-2286.687,235.3928;Inherit;False;Multiply;True;3;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;321;-3530.195,1105.588;Inherit;True;Simplex2D;True;False;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;341;-2063.69,192.9959;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;205;-981.9708,-551.142;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ClampOpNode;281;-1462.831,-506.917;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;335;-1891.659,192.2289;Inherit;False;Property;_CUSTOMFLOWERSCOLOR;CUSTOM FLOWERS COLOR;7;0;Create;True;0;0;0;False;0;False;0;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;398;-1253.63,-473.4861;Inherit;False;FLOWERMASKINVERT;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;322;-3290.813,1101.18;Inherit;True;2;0;FLOAT;0;False;1;FLOAT;0.5;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;402;-707.7606,-549.82;Inherit;False;GRADIENTMASK;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;397;-1839.379,316.5106;Inherit;False;395;FLOWERMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;323;-3209.617,1336.884;Inherit;False;Property;_WindStrength;Wind Strength;21;0;Create;True;0;0;0;False;0;False;0.3;0.203;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;399;-1636.388,377.6237;Inherit;False;398;FLOWERMASKINVERT;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;362;-1548.145,204.0237;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;403;-1583.987,113.7493;Inherit;False;402;GRADIENTMASK;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleDivideOpNode;404;-3084.184,1189.957;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;324;-2914.047,1113.466;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.BlendOpsNode;232;-1320.564,156.1248;Inherit;True;Overlay;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;337;-1424.719,538.6264;Inherit;False;852.152;316.5043;LEAVES CUTOFF;6;289;290;287;288;383;384;;1,1,1,1;0;0
Node;AmplifyShaderEditor.ToggleSwitchNode;357;-1026.966,134.2425;Inherit;False;Property;_CUSTOMCOLORSTINTING;CUSTOM COLORS  TINTING;1;0;Create;True;0;0;0;False;0;False;1;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;325;-2220.044,1312.439;Inherit;False;Constant;_Float4;Float 4;7;0;Create;True;0;0;0;False;0;False;2;0;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;326;-2588.611,968.8419;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldSpaceCameraPos;327;-1764.243,1317.856;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.RegisterLocalVarNode;400;-754.7325,147.8213;Inherit;False;FINALCOLOR;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;287;-1374.72,740.1312;Inherit;False;Property;_LeavesThickness;Leaves Thickness;6;0;Create;True;0;0;0;False;0;False;0.5;0.95;0.1;0.95;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;386;-1618.438,1630.81;Inherit;False;1092.364;358.1904;Comment;5;391;390;389;388;387;TRANSLUCENCY;1,1,1,1;0;0
Node;AmplifyShaderEditor.DynamicAppendNode;329;-2299.171,988.7838;Inherit;True;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;1;False;1;FLOAT4;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;328;-2060.268,1009.169;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;382;-3079.823,327.4753;Inherit;False;ALPHA;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldToObjectTransfNode;331;-1481.877,1302.179;Inherit;False;1;0;FLOAT4;0,0,0,1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;330;-1779.816,982.6949;Inherit;True;3;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0,0,0,0;False;2;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;388;-1542.833,1648.591;Inherit;True;400;FINALCOLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;383;-1397.227,559.4463;Inherit;False;382;ALPHA;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;288;-1252.352,631.116;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;387;-1502.006,1869.382;Inherit;False;Constant;_Float6;Float 6;15;0;Create;True;0;0;0;False;0;False;1;1;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.StepOpNode;289;-1083.13,590.0895;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;389;-1268.077,1726.153;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;332;-1304.967,985.5602;Inherit;True;2;0;FLOAT4;0,0,0,0;False;1;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.OneMinusNode;290;-963.0705,592.4386;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.StaticSwitch;390;-1103.355,1697.202;Inherit;True;Property;_TRANSLUCENCYONOFF;TRANSLUCENCY ON/OFF;10;0;Create;True;0;0;0;False;0;False;0;1;1;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StaticSwitch;333;-1083.237,958.1576;Inherit;False;Property;_CUSTOMWIND;CUSTOM WIND;18;0;Create;True;0;0;0;False;0;False;0;1;1;True;;Toggle;2;Key0;Key1;Create;False;True;All;9;1;FLOAT4;0,0,0,0;False;0;FLOAT4;0,0,0,0;False;2;FLOAT4;0,0,0,0;False;3;FLOAT4;0,0,0,0;False;4;FLOAT4;0,0,0,0;False;5;FLOAT4;0,0,0,0;False;6;FLOAT4;0,0,0,0;False;7;FLOAT4;0,0,0,0;False;8;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;391;-795.5361,1696.407;Inherit;False;TRANSLUCENCY;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;384;-794.2621,590.0046;Inherit;False;TRANSPARENCY;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;393;-813.6538,966.9396;Inherit;False;WIND;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;385;388.6036,314.7903;Inherit;False;384;TRANSPARENCY;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;392;390.877,240.8879;Inherit;False;391;TRANSLUCENCY;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;394;406.6841,386.0674;Inherit;False;393;WIND;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;401;244.2854,64.10895;Inherit;False;400;FINALCOLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;130;944.7643,-169.6443;Inherit;False;Property;_MaskClipValue;Mask Clip Value;8;1;[HideInInspector];Fetch;True;0;0;0;False;0;False;0.5;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;407;256.6289,164.2671;Inherit;False;Constant;_Float5;Float 5;16;0;Create;True;0;0;0;False;0;False;0;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;62;648.3509,76.09714;Float;False;True;-1;2;;0;0;Standard;Polytope Studio/PT_Vegetation_Flowers_Shader;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;True;False;False;False;False;Off;0;False;;0;False;;False;0;False;;0;False;;False;0;Custom;0.5;True;True;0;True;TransparentCutout;;Geometry;ForwardOnly;18;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;False;2;15;10;25;False;0.5;True;0;0;False;;0;False;;0;0;False;;0;False;;0;False;;0;False;;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Absolute;0;;-1;11;-1;-1;0;False;0;0;False;;-1;0;True;_MaskClipValue;1;Pragma;multi_compile __ LOD_FADE_CROSSFADE;False;;Custom;0;0;False;0.1;False;;0;False;;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
WireConnection;235;1;233;0
WireConnection;204;1;206;0
WireConnection;236;0;235;2
WireConnection;236;1;234;0
WireConnection;207;0;204;1
WireConnection;207;1;208;0
WireConnection;238;0;207;0
WireConnection;238;1;236;0
WireConnection;348;0;343;2
WireConnection;209;0;238;0
WireConnection;345;0;348;0
WireConnection;345;1;344;0
WireConnection;406;0;345;0
WireConnection;406;1;405;0
WireConnection;248;0;209;0
WireConnection;395;0;248;0
WireConnection;354;0;406;0
WireConnection;2;0;127;0
WireConnection;313;0;2;0
WireConnection;356;0;352;0
WireConnection;356;1;355;0
WireConnection;356;2;354;0
WireConnection;317;0;316;0
WireConnection;317;1;315;0
WireConnection;336;0;313;0
WireConnection;336;1;396;0
WireConnection;380;0;356;0
WireConnection;319;0;318;0
WireConnection;319;2;317;0
WireConnection;310;0;308;0
WireConnection;310;1;336;0
WireConnection;310;2;396;0
WireConnection;321;0;319;0
WireConnection;321;1;320;0
WireConnection;341;0;2;0
WireConnection;341;1;310;0
WireConnection;341;2;396;0
WireConnection;205;0;381;0
WireConnection;205;1;238;0
WireConnection;205;2;395;0
WireConnection;281;0;238;0
WireConnection;335;0;2;0
WireConnection;335;1;341;0
WireConnection;398;0;281;0
WireConnection;322;0;321;0
WireConnection;402;0;205;0
WireConnection;362;0;313;0
WireConnection;362;1;335;0
WireConnection;362;2;397;0
WireConnection;404;0;322;0
WireConnection;324;0;404;0
WireConnection;324;1;323;0
WireConnection;232;0;403;0
WireConnection;232;1;362;0
WireConnection;232;2;399;0
WireConnection;357;0;335;0
WireConnection;357;1;232;0
WireConnection;326;0;324;0
WireConnection;326;1;318;1
WireConnection;400;0;357;0
WireConnection;329;0;326;0
WireConnection;329;1;318;2
WireConnection;329;2;318;3
WireConnection;328;0;318;2
WireConnection;328;1;325;0
WireConnection;382;0;2;4
WireConnection;331;0;327;0
WireConnection;330;0;318;0
WireConnection;330;1;329;0
WireConnection;330;2;328;0
WireConnection;288;0;287;0
WireConnection;289;0;383;0
WireConnection;289;1;288;0
WireConnection;389;0;388;0
WireConnection;389;1;387;0
WireConnection;332;0;330;0
WireConnection;332;1;331;4
WireConnection;290;0;289;0
WireConnection;390;0;389;0
WireConnection;333;1;318;0
WireConnection;333;0;332;0
WireConnection;391;0;390;0
WireConnection;384;0;290;0
WireConnection;393;0;333;0
WireConnection;62;0;401;0
WireConnection;62;3;407;0
WireConnection;62;4;407;0
WireConnection;62;7;392;0
WireConnection;62;10;385;0
WireConnection;62;11;394;0
ASEEND*/
//CHKSM=0A765D0DB346C969F8E6E8F50F3E290B7B3E9163