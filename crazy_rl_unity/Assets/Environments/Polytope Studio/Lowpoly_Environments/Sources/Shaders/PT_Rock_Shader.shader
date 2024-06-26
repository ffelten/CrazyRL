// Made with Amplify Shader Editor v1.9.1.5
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/PT_Rock_Shader"
{
	Properties
	{
		[NoScaleOffset]_BaseTexture("Base Texture", 2D) = "white" {}
		_Smoothness("Smoothness", Range( 0 , 1)) = 0.2
		[Toggle(_GRADIENTONOFF_ON)] _GRADIENTONOFF("GRADIENT  ON/OFF", Float) = 0
		[HDR]_TopColor("Top Color", Color) = (0.4811321,0.4036026,0.2382966,1)
		[HDR]_GroundColor("Ground Color", Color) = (0.08490568,0.05234205,0.04846032,1)
		[HideInInspector]_SnowDirection("Snow Direction", Vector) = (0.1,1,0.1,0)
		_Gradient("Gradient ", Range( 0 , 1)) = 1
		_GradientPower("Gradient Power", Range( 0 , 10)) = 1
		[Toggle]_WorldObjectGradient("World/Object Gradient", Float) = 1
		[Toggle(_DECALSONOFF_ON)] _DECALSONOFF("DECALS ON/OFF", Float) = 0
		[NoScaleOffset]_DecalsTexture("Decals Texture", 2D) = "white" {}
		_DecalsColor("Decals Color", Color) = (0,0,0,0)
		[Toggle]_DECALEMISSIONONOFF("DECAL EMISSION ON/OFF", Float) = 1
		[HDR]_DecakEmissionColor("Decak Emission Color", Color) = (1,0.9248579,0,0)
		_DecalEmissionIntensity("Decal Emission Intensity", Range( 0 , 10)) = 4
		[Toggle]_ANIMATEDECALEMISSIONONOFF("ANIMATE DECAL EMISSION ON/OFF", Float) = 1
		[Toggle(_DETAILTEXTUREONOFF_ON)] _DETAILTEXTUREONOFF("DETAIL TEXTURE  ON/OFF", Float) = 0
		[NoScaleOffset]_DetailTexture("Detail Texture", 2D) = "white" {}
		_DetailTextureTiling("Detail Texture Tiling", Range( 0.1 , 10)) = 0.5
		[Toggle(_SNOWONOFF_ON)] _SNOWONOFF("SNOW ON/OFF", Float) = 0
		_SnowCoverage("Snow Coverage", Range( 0 , 1)) = 0.46
		_SnowAmount("Snow Amount", Range( 0 , 1)) = 1
		_SnowFade("Snow Fade", Range( 0 , 1)) = 0.32
		[Toggle]_SnowNoiseOnOff("Snow Noise On/Off", Float) = 1
		_SnowNoiseScale("Snow Noise Scale", Range( 0 , 100)) = 87.23351
		_SnowNoiseContrast("Snow Noise Contrast", Range( 0 , 1)) = 0.002
		[HideInInspector]_Vector1("Vector 1", Vector) = (0,1,0,0)
		[Toggle(_TOPPROJECTIONONOFF_ON)] _TOPPROJECTIONONOFF("TOP PROJECTION ON/OFF", Float) = 0
		[NoScaleOffset]_TopProjectionTexture("Top Projection Texture", 2D) = "white" {}
		_TopProjectionTextureTiling("Top Projection Texture Tiling", Range( 0.1 , 10)) = 0.5
		_TopProjectionTextureCoverage("Top Projection Texture  Coverage", Range( 0 , 1)) = 1
		[HDR]_OreColor("Ore Color", Color) = (1,0.9248579,0,0)
		[Toggle]_OREEMISSIONONOFF("ORE EMISSION ON/OFF", Float) = 0
		[HDR]_OreEmissionColor("Ore Emission Color", Color) = (1,0.9248579,0,0)
		_OreEmissionIntensity("Ore Emission Intensity", Range( 0 , 10)) = 1
		[Toggle]_ANIMATEOREEMISSIONONOFF("ANIMATE ORE  EMISSION ON/OFF", Float) = 0
		[HideInInspector] _texcoord2( "", 2D ) = "white" {}
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "Opaque"  "Queue" = "Geometry+0" "IsEmissive" = "true"  }
		Cull Off
		CGINCLUDE
		#include "UnityShaderVariables.cginc"
		#include "UnityPBSLighting.cginc"
		#include "Lighting.cginc"
		#pragma target 3.5
		#pragma shader_feature_local _SNOWONOFF_ON
		#pragma shader_feature_local _TOPPROJECTIONONOFF_ON
		#pragma shader_feature_local _DECALSONOFF_ON
		#pragma shader_feature_local _DETAILTEXTUREONOFF_ON
		#pragma shader_feature_local _GRADIENTONOFF_ON
		#ifdef UNITY_PASS_SHADOWCASTER
			#undef INTERNAL_DATA
			#undef WorldReflectionVector
			#undef WorldNormalVector
			#define INTERNAL_DATA half3 internalSurfaceTtoW0; half3 internalSurfaceTtoW1; half3 internalSurfaceTtoW2;
			#define WorldReflectionVector(data,normal) reflect (data.worldRefl, half3(dot(data.internalSurfaceTtoW0,normal), dot(data.internalSurfaceTtoW1,normal), dot(data.internalSurfaceTtoW2,normal)))
			#define WorldNormalVector(data,normal) half3(dot(data.internalSurfaceTtoW0,normal), dot(data.internalSurfaceTtoW1,normal), dot(data.internalSurfaceTtoW2,normal))
		#endif
		struct Input
		{
			float2 uv_texcoord;
			float3 worldPos;
			float3 worldNormal;
			INTERNAL_DATA
			float2 uv2_texcoord2;
			float4 vertexColor : COLOR;
		};

		uniform sampler2D _BaseTexture;
		uniform float4 _GroundColor;
		uniform float4 _TopColor;
		uniform float _WorldObjectGradient;
		uniform float _Gradient;
		uniform float _GradientPower;
		uniform sampler2D _DetailTexture;
		uniform float _DetailTextureTiling;
		uniform float4 _DecalsColor;
		uniform sampler2D _DecalsTexture;
		uniform sampler2D _TopProjectionTexture;
		uniform float _TopProjectionTextureTiling;
		uniform float3 _Vector1;
		uniform float _TopProjectionTextureCoverage;
		uniform float _SnowNoiseOnOff;
		uniform float _SnowAmount;
		uniform float _SnowFade;
		uniform float _SnowCoverage;
		uniform float3 _SnowDirection;
		uniform float _SnowNoiseScale;
		uniform float _SnowNoiseContrast;
		uniform float4 _OreColor;
		uniform float _DECALEMISSIONONOFF;
		uniform float _DecalEmissionIntensity;
		uniform float _ANIMATEDECALEMISSIONONOFF;
		uniform float4 _DecakEmissionColor;
		uniform float _OREEMISSIONONOFF;
		uniform float _OreEmissionIntensity;
		uniform float _ANIMATEOREEMISSIONONOFF;
		uniform float4 _OreEmissionColor;
		uniform float _Smoothness;


		inline float4 TriplanarSampling173( sampler2D topTexMap, sampler2D midTexMap, sampler2D botTexMap, float3 worldPos, float3 worldNormal, float falloff, float2 tiling, float3 normalScale, float3 index )
		{
			float3 projNormal = ( pow( abs( worldNormal ), falloff ) );
			projNormal /= ( projNormal.x + projNormal.y + projNormal.z ) + 0.00001;
			float3 nsign = sign( worldNormal );
			float negProjNormalY = max( 0, projNormal.y * -nsign.y );
			projNormal.y = max( 0, projNormal.y * nsign.y );
			half4 xNorm; half4 yNorm; half4 yNormN; half4 zNorm;
			xNorm  = tex2D( midTexMap, tiling * worldPos.zy * float2(  nsign.x, 1.0 ) );
			yNorm  = tex2D( topTexMap, tiling * worldPos.xz * float2(  nsign.y, 1.0 ) );
			yNormN = tex2D( botTexMap, tiling * worldPos.xz * float2(  nsign.y, 1.0 ) );
			zNorm  = tex2D( midTexMap, tiling * worldPos.xy * float2( -nsign.z, 1.0 ) );
			return xNorm * projNormal.x + yNorm * projNormal.y + yNormN * negProjNormalY + zNorm * projNormal.z;
		}


		inline float4 TriplanarSampling525( sampler2D topTexMap, float3 worldPos, float3 worldNormal, float falloff, float2 tiling, float3 normalScale, float3 index )
		{
			float3 projNormal = ( pow( abs( worldNormal ), falloff ) );
			projNormal /= ( projNormal.x + projNormal.y + projNormal.z ) + 0.00001;
			float3 nsign = sign( worldNormal );
			half4 xNorm; half4 yNorm; half4 zNorm;
			xNorm = tex2D( topTexMap, tiling * worldPos.zy * float2(  nsign.x, 1.0 ) );
			yNorm = tex2D( topTexMap, tiling * worldPos.xz * float2(  nsign.y, 1.0 ) );
			zNorm = tex2D( topTexMap, tiling * worldPos.xy * float2( -nsign.z, 1.0 ) );
			return xNorm * projNormal.x + yNorm * projNormal.y + zNorm * projNormal.z;
		}


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


		void surf( Input i , inout SurfaceOutputStandard o )
		{
			o.Normal = float3(0,0,1);
			float2 uv_BaseTexture490 = i.uv_texcoord;
			float4 BASETEXTURE498 = tex2D( _BaseTexture, uv_BaseTexture490 );
			float3 ase_worldPos = i.worldPos;
			float4 ase_vertex4Pos = mul( unity_WorldToObject, float4( i.worldPos , 1 ) );
			float clampResult627 = clamp( pow( ( ( (( _WorldObjectGradient )?( ase_vertex4Pos.y ):( ase_worldPos.y )) + 1.5 ) * _Gradient ) , _GradientPower ) , -1.0 , 1.0 );
			float4 lerpResult629 = lerp( _GroundColor , _TopColor , clampResult627);
			float4 Gradient630 = lerpResult629;
			float4 color644 = IsGammaSpace() ? float4(0.8962264,0.8962264,0.8962264,0) : float4(0.7799658,0.7799658,0.7799658,0);
			float4 blendOpSrc643 = BASETEXTURE498;
			float4 blendOpDest643 = color644;
			#ifdef _GRADIENTONOFF_ON
				float4 staticSwitch634 = ( Gradient630 * ( saturate(  (( blendOpSrc643 > 0.5 ) ? ( 1.0 - ( 1.0 - 2.0 * ( blendOpSrc643 - 0.5 ) ) * ( 1.0 - blendOpDest643 ) ) : ( 2.0 * blendOpSrc643 * blendOpDest643 ) ) )) );
			#else
				float4 staticSwitch634 = BASETEXTURE498;
			#endif
			float2 temp_cast_0 = (_DetailTextureTiling).xx;
			float3 ase_worldNormal = WorldNormalVector( i, float3( 0, 0, 1 ) );
			float4 triplanar173 = TriplanarSampling173( _DetailTexture, _DetailTexture, _DetailTexture, ase_worldPos, ase_worldNormal, 1.0, temp_cast_0, float3( 1,1,1 ), float3(0,0,0) );
			float4 DETAILTEXTUREvar414 = triplanar173;
			#ifdef _DETAILTEXTUREONOFF_ON
				float4 staticSwitch543 = ( DETAILTEXTUREvar414 * staticSwitch634 );
			#else
				float4 staticSwitch543 = staticSwitch634;
			#endif
			float4 decalscolor730 = _DecalsColor;
			float2 uv1_DecalsTexture495 = i.uv2_texcoord2;
			float DECALSMASK497 = tex2D( _DecalsTexture, uv1_DecalsTexture495 ).a;
			float4 lerpResult596 = lerp( staticSwitch543 , decalscolor730 , DECALSMASK497);
			#ifdef _DECALSONOFF_ON
				float4 staticSwitch600 = lerpResult596;
			#else
				float4 staticSwitch600 = staticSwitch543;
			#endif
			float2 temp_cast_4 = (_TopProjectionTextureTiling).xx;
			float4 triplanar525 = TriplanarSampling525( _TopProjectionTexture, ase_worldPos, ase_worldNormal, 1.0, temp_cast_4, 1.0, 0 );
			float4 TOPPROJECTION527 = triplanar525;
			float dotResult524 = dot( ase_worldNormal , _Vector1 );
			float saferPower582 = abs( ( ( dotResult524 * _TopProjectionTextureCoverage ) * 3.0 ) );
			float clampResult584 = clamp( pow( saferPower582 , 5.0 ) , 0.0 , 1.0 );
			float TOPPROJECTIONMASK528 = clampResult584;
			float4 lerpResult555 = lerp( staticSwitch600 , TOPPROJECTION527 , TOPPROJECTIONMASK528);
			#ifdef _TOPPROJECTIONONOFF_ON
				float4 staticSwitch557 = lerpResult555;
			#else
				float4 staticSwitch557 = staticSwitch600;
			#endif
			float4 color205 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			float dotResult211 = dot( ase_worldNormal , _SnowDirection );
			float smoothstepResult552 = smoothstep( 0.0 , _SnowFade , ( (-1.0 + (_SnowCoverage - 0.0) * (1.0 - -1.0) / (1.0 - 0.0)) + dotResult211 ));
			float4 temp_output_363_0 = ( ( (0.0 + (_SnowAmount - 0.0) * (10.0 - 0.0) / (1.0 - 0.0)) * color205 ) * smoothstepResult552 );
			float4 transform200 = mul(unity_WorldToObject,float4( ase_worldPos , 0.0 ));
			float4 appendResult209 = (float4(transform200.x , transform200.z , 0.0 , 0.0));
			float simplePerlin2D213 = snoise( appendResult209.xy*_SnowNoiseScale );
			simplePerlin2D213 = simplePerlin2D213*0.5 + 0.5;
			float saferPower216 = abs( simplePerlin2D213 );
			float4 SNOW220 = (( _SnowNoiseOnOff )?( ( pow( saferPower216 , _SnowNoiseContrast ) * temp_output_363_0 ) ):( temp_output_363_0 ));
			#ifdef _SNOWONOFF_ON
				float4 staticSwitch545 = ( staticSwitch557 + SNOW220 );
			#else
				float4 staticSwitch545 = staticSwitch557;
			#endif
			float4 lerpResult607 = lerp( staticSwitch545 , _OreColor , ( 1.0 - i.vertexColor.a ));
			float4 COLOR539 = lerpResult607;
			o.Albedo = COLOR539.rgb;
			float3 temp_cast_9 = (1.0).xxx;
			float4 color717 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			float4 color716 = IsGammaSpace() ? float4(0,0,0,0) : float4(0,0,0,0);
			float4 lerpResult718 = lerp( color717 , color716 , (_SinTime.w*0.3 + 0.5));
			float3 desaturateInitialColor720 = lerpResult718.rgb;
			float desaturateDot720 = dot( desaturateInitialColor720, float3( 0.299, 0.587, 0.114 ));
			float3 desaturateVar720 = lerp( desaturateInitialColor720, desaturateDot720.xxx, 1.0 );
			float4 Decalemission685 = (( _DECALEMISSIONONOFF )?( ( ( float4( ( _DecalEmissionIntensity * (( _ANIMATEDECALEMISSIONONOFF )?( desaturateVar720 ):( temp_cast_9 )) ) , 0.0 ) * _DecakEmissionColor ) * DECALSMASK497 ) ):( float4( 0,0,0,0 ) ));
			float3 temp_cast_12 = (0.1).xxx;
			float4 color701 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			float4 color702 = IsGammaSpace() ? float4(0,0,0,0) : float4(0,0,0,0);
			float4 lerpResult703 = lerp( color701 , color702 , (_SinTime.w*0.3 + 0.5));
			float3 desaturateInitialColor704 = lerpResult703.rgb;
			float desaturateDot704 = dot( desaturateInitialColor704, float3( 0.299, 0.587, 0.114 ));
			float3 desaturateVar704 = lerp( desaturateInitialColor704, desaturateDot704.xxx, 1.0 );
			float4 oreemission684 = (( _OREEMISSIONONOFF )?( ( ( float4( ( _OreEmissionIntensity * (( _ANIMATEOREEMISSIONONOFF )?( desaturateVar704 ):( temp_cast_12 )) ) , 0.0 ) * _OreEmissionColor ) * ( 1.0 - i.vertexColor.a ) ) ):( float4( 0,0,0,0 ) ));
			o.Emission = ( Decalemission685 + oreemission684 ).rgb;
			float4 color617 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			o.Smoothness = ( _Smoothness * color617 ).r;
			o.Alpha = 1;
		}

		ENDCG
		CGPROGRAM
		#pragma surface surf Standard keepalpha fullforwardshadows dithercrossfade 

		ENDCG
		Pass
		{
			Name "ShadowCaster"
			Tags{ "LightMode" = "ShadowCaster" }
			ZWrite On
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 3.5
			#pragma multi_compile_shadowcaster
			#pragma multi_compile UNITY_PASS_SHADOWCASTER
			#pragma skip_variants FOG_LINEAR FOG_EXP FOG_EXP2
			#include "HLSLSupport.cginc"
			#if ( SHADER_API_D3D11 || SHADER_API_GLCORE || SHADER_API_GLES || SHADER_API_GLES3 || SHADER_API_METAL || SHADER_API_VULKAN )
				#define CAN_SKIP_VPOS
			#endif
			#include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "UnityPBSLighting.cginc"
			struct v2f
			{
				V2F_SHADOW_CASTER;
				float4 customPack1 : TEXCOORD1;
				float4 tSpace0 : TEXCOORD2;
				float4 tSpace1 : TEXCOORD3;
				float4 tSpace2 : TEXCOORD4;
				half4 color : COLOR0;
				UNITY_VERTEX_INPUT_INSTANCE_ID
				UNITY_VERTEX_OUTPUT_STEREO
			};
			v2f vert( appdata_full v )
			{
				v2f o;
				UNITY_SETUP_INSTANCE_ID( v );
				UNITY_INITIALIZE_OUTPUT( v2f, o );
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO( o );
				UNITY_TRANSFER_INSTANCE_ID( v, o );
				Input customInputData;
				float3 worldPos = mul( unity_ObjectToWorld, v.vertex ).xyz;
				half3 worldNormal = UnityObjectToWorldNormal( v.normal );
				half3 worldTangent = UnityObjectToWorldDir( v.tangent.xyz );
				half tangentSign = v.tangent.w * unity_WorldTransformParams.w;
				half3 worldBinormal = cross( worldNormal, worldTangent ) * tangentSign;
				o.tSpace0 = float4( worldTangent.x, worldBinormal.x, worldNormal.x, worldPos.x );
				o.tSpace1 = float4( worldTangent.y, worldBinormal.y, worldNormal.y, worldPos.y );
				o.tSpace2 = float4( worldTangent.z, worldBinormal.z, worldNormal.z, worldPos.z );
				o.customPack1.xy = customInputData.uv_texcoord;
				o.customPack1.xy = v.texcoord;
				o.customPack1.zw = customInputData.uv2_texcoord2;
				o.customPack1.zw = v.texcoord1;
				TRANSFER_SHADOW_CASTER_NORMALOFFSET( o )
				o.color = v.color;
				return o;
			}
			half4 frag( v2f IN
			#if !defined( CAN_SKIP_VPOS )
			, UNITY_VPOS_TYPE vpos : VPOS
			#endif
			) : SV_Target
			{
				UNITY_SETUP_INSTANCE_ID( IN );
				Input surfIN;
				UNITY_INITIALIZE_OUTPUT( Input, surfIN );
				surfIN.uv_texcoord = IN.customPack1.xy;
				surfIN.uv2_texcoord2 = IN.customPack1.zw;
				float3 worldPos = float3( IN.tSpace0.w, IN.tSpace1.w, IN.tSpace2.w );
				half3 worldViewDir = normalize( UnityWorldSpaceViewDir( worldPos ) );
				surfIN.worldPos = worldPos;
				surfIN.worldNormal = float3( IN.tSpace0.z, IN.tSpace1.z, IN.tSpace2.z );
				surfIN.internalSurfaceTtoW0 = IN.tSpace0.xyz;
				surfIN.internalSurfaceTtoW1 = IN.tSpace1.xyz;
				surfIN.internalSurfaceTtoW2 = IN.tSpace2.xyz;
				surfIN.vertexColor = IN.color;
				SurfaceOutputStandard o;
				UNITY_INITIALIZE_OUTPUT( SurfaceOutputStandard, o )
				surf( surfIN, o );
				#if defined( CAN_SKIP_VPOS )
				float2 vpos = IN.pos;
				#endif
				SHADOW_CASTER_FRAGMENT( IN )
			}
			ENDCG
		}
	}
	Fallback "Diffuse"
}
/*ASEBEGIN
Version=19105
Node;AmplifyShaderEditor.CommentaryNode;619;-3010.418,-275.974;Inherit;False;1754.419;983.1141;GRADIENT;13;629;628;627;626;625;624;623;621;620;640;641;735;736;GRADIENT;1,1,1,1;0;0
Node;AmplifyShaderEditor.WorldPosInputsNode;640;-3043.074,-63.0602;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.PosVertexDataNode;620;-3073.949,220.5544;Inherit;False;1;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;735;-2823.862,296.1774;Inherit;False;Constant;_Float8;Float 8;34;0;Create;True;0;0;0;False;0;False;1.5;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;641;-2881.464,109.3181;Inherit;False;Property;_WorldObjectGradient;World/Object Gradient;8;0;Create;True;0;0;0;False;0;False;1;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;621;-2981.692,412.0216;Float;False;Property;_Gradient;Gradient ;6;0;Create;True;0;0;0;False;0;False;1;0.443;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;736;-2604.904,129.8654;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;623;-2494.163,207.1993;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;624;-2639.132,381.3199;Inherit;False;Property;_GradientPower;Gradient Power;7;0;Create;True;0;0;0;False;0;False;1;1.68;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;535;-1653.702,1225.264;Inherit;False;884.7478;315.1912;Comment;3;491;490;498;BASE TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.PowerNode;625;-2153.887,207.9514;Inherit;True;False;2;0;FLOAT;0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;491;-1603.702,1310.455;Inherit;True;Property;_BaseTexture;Base Texture;0;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;a195a893970f3ab4990bd06e18a0b308;a195a893970f3ab4990bd06e18a0b308;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.ColorNode;626;-2262.027,-42.2462;Float;False;Property;_TopColor;Top Color;3;1;[HDR];Create;True;0;0;0;False;0;False;0.4811321,0.4036026,0.2382966,1;0.8962264,0.8962264,0.8962264,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SamplerNode;490;-1355.477,1309.809;Inherit;True;Property;_TextureSample0;Texture Sample 0;17;0;Create;True;0;0;0;False;0;False;-1;None;None;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ClampOpNode;627;-1977.984,164.9912;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;-1;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;628;-2265.061,-228.3453;Float;False;Property;_GroundColor;Ground Color;4;1;[HDR];Create;True;0;0;0;False;0;False;0.08490568,0.05234205,0.04846032,1;0.1415094,0.08937437,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;629;-1733.621,-127.7129;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;461;-3021.441,3894.707;Inherit;False;1516.12;1028.51;Comment;5;414;173;175;273;183;DETAIL TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;517;-3072.775,5107.095;Inherit;False;2347.7;1068.11;Comment;16;528;527;526;525;524;523;522;521;520;519;518;581;582;584;737;738;TOP PROJECTION;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;498;-992.9542,1275.263;Inherit;False;BASETEXTURE;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;644;-255.1736,1682.719;Inherit;False;Constant;_Color2;Color 2;27;0;Create;True;0;0;0;False;0;False;0.8962264,0.8962264,0.8962264,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.CommentaryNode;198;-2825.285,2251.937;Inherit;False;2125.222;1412.266;Comment;23;219;218;216;215;214;213;211;209;208;207;204;203;201;200;199;363;202;210;205;552;553;554;220;SNOW;1,1,1,1;0;0
Node;AmplifyShaderEditor.GetLocalVarNode;541;-277.029,1506.594;Inherit;False;498;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.WorldPosInputsNode;273;-2970.759,4352.548;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.TexturePropertyNode;175;-3004,4144.234;Inherit;True;Property;_DetailTexture;Detail Texture;17;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.RangedFloatNode;183;-2962.298,4517.901;Inherit;False;Property;_DetailTextureTiling;Detail Texture Tiling;18;0;Create;True;0;0;0;False;0;False;0.5;0.31;0.1;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;630;-1044.418,-144.1897;Inherit;False;Gradient;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.BlendOpsNode;643;-50.17358,1639.719;Inherit;False;HardLight;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.TriplanarNode;173;-2450.177,4173.919;Inherit;True;Cylindrical;World;False;Top Texture 1;_TopTexture1;white;1;None;Mid Texture 1;_MidTexture1;white;4;None;Bot Texture 1;_BotTexture1;white;2;None;Triplanar Sampler;Tangent;10;0;SAMPLER2D;;False;5;FLOAT;1;False;1;SAMPLER2D;;False;6;FLOAT;0;False;2;SAMPLER2D;;False;7;FLOAT;0;False;9;FLOAT3;0,0,0;False;8;FLOAT3;1,1,1;False;3;FLOAT2;1,1;False;4;FLOAT;1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.Vector3Node;204;-2597.414,3354.151;Inherit;False;Property;_SnowDirection;Snow Direction;5;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.1,1,0.1;0.1,1,0.1;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.CommentaryNode;534;-2474.03,1642.298;Inherit;False;1745.965;569.649;Comment;5;496;495;497;729;730;DECALS;1,1,1,1;0;0
Node;AmplifyShaderEditor.WorldPosInputsNode;199;-2764.211,2294.017;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.GetLocalVarNode;632;-116.1384,1895.9;Inherit;False;630;Gradient;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;202;-2586.97,2997.915;Inherit;False;Property;_SnowCoverage;Snow Coverage;20;0;Create;True;0;0;0;False;0;False;0.46;0.23;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;210;-2294.394,3003.907;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;-1;False;4;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldToObjectTransfNode;200;-2574.372,2292.027;Inherit;False;1;0;FLOAT4;0,0,0,1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RegisterLocalVarNode;414;-1979.137,4178.824;Inherit;True;DETAILTEXTUREvar;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;203;-2805.705,2607.053;Inherit;False;Property;_SnowAmount;Snow Amount;21;0;Create;True;0;0;0;False;0;False;1;0.215;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.DotProductOpNode;211;-2327.456,3227.93;Inherit;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;496;-2424.03,1692.298;Inherit;True;Property;_DecalsTexture;Decals Texture;10;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;d294e9544b9eca64188ea9d2482ea8a1;d294e9544b9eca64188ea9d2482ea8a1;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.CommentaryNode;692;-5490.614,-272.6373;Inherit;False;2275.294;729.1099;Comment;20;698;704;684;677;676;663;673;708;662;664;707;659;723;705;703;699;702;700;701;727;ORE EMISSION;1,1,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;693;-5736.309,1188.724;Inherit;False;2615.817;1052.083;Comment;19;720;718;719;715;717;716;714;713;721;679;722;681;680;683;686;688;685;726;728;DECAL EMISSION;1,1,1,1;0;0
Node;AmplifyShaderEditor.DynamicAppendNode;209;-2293.907,2326.596;Inherit;False;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;207;-2420.321,2510.848;Inherit;False;Property;_SnowNoiseScale;Snow Noise Scale;24;0;Create;True;0;0;0;False;0;False;87.23351;4.9;0;100;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;726;-5694.521,2042.805;Inherit;False;Constant;_Float5;Float 5;36;0;Create;True;0;0;0;False;0;False;0.5;0;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;553;-2205.344,3532.128;Inherit;False;Property;_SnowFade;Snow Fade;22;0;Create;True;0;0;0;False;0;False;0.32;0.736;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;727;-5416.934,394.9289;Inherit;False;Constant;_Float6;Float 6;37;0;Create;True;0;0;0;False;0;False;0.5;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;713;-5710.042,1941.29;Inherit;False;Constant;_Float2;Float 2;33;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;729;-1895.781,2042.536;Inherit;False;Property;_DecalsColor;Decals Color;11;0;Create;True;0;0;0;False;0;False;0,0,0,0;0.008939266,0,1,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SamplerNode;495;-2075.804,1699.652;Inherit;True;Property;_TextureSample1;Texture Sample 1;17;0;Create;True;0;0;0;False;0;False;-1;None;None;True;1;False;white;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;205;-2533.157,2799.42;Inherit;False;Constant;_Color1;Color 1;30;0;Create;True;0;0;0;False;0;False;1,1,1,0;1,1,1,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.GetLocalVarNode;542;407.481,1692.665;Inherit;False;414;DETAILTEXTUREvar;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;700;-5417.658,315.0158;Inherit;False;Constant;_Float0;Float 0;33;0;Create;True;0;0;0;False;0;False;0.3;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;208;-2507.276,2613.048;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0;False;4;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.StaticSwitch;634;312.4376,1501.704;Inherit;False;Property;_GRADIENTONOFF;GRADIENT  ON/OFF;2;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleAddOpNode;554;-2058.173,3090.27;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SinTimeNode;714;-5698.453,1748.684;Inherit;False;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SinTimeNode;698;-5393.771,152.6686;Inherit;False;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;215;-2235.493,2708.072;Inherit;False;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;213;-2065.902,2319.317;Inherit;True;Simplex2D;True;False;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;730;-1542.781,2044.536;Inherit;False;decalscolor;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;701;-5434.534,-200.4745;Inherit;False;Constant;_Color3;Color 3;33;0;Create;True;0;0;0;False;0;False;1,1,1,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RegisterLocalVarNode;497;-1729.211,1753.205;Inherit;True;DECALSMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ScaleAndOffsetNode;699;-5229.359,252.6765;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;1;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;716;-5709.142,1563.391;Inherit;False;Constant;_Color5;Color 5;33;0;Create;True;0;0;0;False;0;False;0,0,0,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ScaleAndOffsetNode;715;-5552.341,1836.492;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;1;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SmoothstepOpNode;552;-1779.915,3194.36;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;702;-5413.459,-25.62411;Inherit;False;Constant;_Color4;Color 4;33;0;Create;True;0;0;0;False;0;False;0,0,0,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;717;-5711.142,1391.391;Inherit;False;Constant;_Color6;Color 6;33;0;Create;True;0;0;0;False;0;False;1,1,1,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.WorldPosInputsNode;520;-3004.478,5377.535;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;612;661.0915,1604.322;Inherit;False;2;2;0;FLOAT4;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;214;-2063.839,2558.468;Inherit;False;Property;_SnowNoiseContrast;Snow Noise Contrast;25;0;Create;True;0;0;0;False;0;False;0.002;0.223;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;705;-5186.848,114.9911;Inherit;False;Constant;_Float1;Float 1;33;0;Create;True;0;0;0;False;0;False;1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;363;-1811.692,2739.948;Inherit;True;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.TriplanarNode;525;-2492.181,5205.906;Inherit;True;Spherical;World;False;Top Texture 0;_TopTexture0;white;1;None;Mid Texture 0;_MidTexture0;white;7;None;Bot Texture 0;_BotTexture0;white;6;None;Triplanar Sampler;Tangent;10;0;SAMPLER2D;;False;5;FLOAT;1;False;1;SAMPLER2D;;False;6;FLOAT;0;False;2;SAMPLER2D;;False;7;FLOAT;0;False;9;FLOAT3;0,0,0;False;8;FLOAT;1;False;3;FLOAT2;1,1;False;4;FLOAT;1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;719;-5349.931,1949.707;Inherit;False;Constant;_Float3;Float 3;33;0;Create;True;0;0;0;False;0;False;1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;703;-5121.776,-48.73544;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;718;-5417.459,1540.279;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;597;846.0334,1697.674;Inherit;False;730;decalscolor;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.StaticSwitch;543;847.7029,1500.852;Inherit;False;Property;_DETAILTEXTUREONOFF;DETAIL TEXTURE  ON/OFF;16;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;598;837.2399,1781.854;Inherit;False;497;DECALSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.PowerNode;216;-1764.981,2326.336;Inherit;False;True;2;0;FLOAT;0;False;1;FLOAT;0.1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;218;-1497.883,2507.298;Inherit;True;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.DesaturateOpNode;720;-5244.955,1679.786;Inherit;False;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RangedFloatNode;723;-4994.15,-107.1673;Inherit;False;Constant;_Float4;Float 4;35;0;Create;True;0;0;0;False;0;False;0.1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;728;-5249.092,1362.56;Inherit;False;Constant;_Float7;Float 7;32;0;Create;True;0;0;0;False;0;False;1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;596;1112.479,1677.352;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.DesaturateOpNode;704;-4948.056,-4.521085;Inherit;True;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;527;-2054.534,5187.752;Inherit;True;TOPPROJECTION;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;528;-1527.419,5672.312;Inherit;True;TOPPROJECTIONMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;707;-4834.949,-112.8421;Inherit;False;Property;_ANIMATEOREEMISSIONONOFF;ANIMATE ORE  EMISSION ON/OFF;35;0;Create;True;0;0;0;False;0;False;0;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.GetLocalVarNode;556;1352.4,1694.833;Inherit;False;527;TOPPROJECTION;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;659;-5047.669,-202.7318;Inherit;False;Property;_OreEmissionIntensity;Ore Emission Intensity;34;0;Create;True;0;0;0;False;0;False;1;3.74;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;721;-5080.235,1357.998;Inherit;False;Property;_ANIMATEDECALEMISSIONONOFF;ANIMATE DECAL EMISSION ON/OFF;15;0;Create;True;0;0;0;False;0;False;1;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.StaticSwitch;600;1300.288,1498.409;Inherit;False;Property;_DECALSONOFF;DECALS ON/OFF;9;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;679;-5090.93,1252.894;Inherit;False;Property;_DecalEmissionIntensity;Decal Emission Intensity;14;0;Create;True;0;0;0;False;0;False;4;5.78;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;680;-4712.388,1357.25;Inherit;False;Property;_DecakEmissionColor;Decak Emission Color;13;1;[HDR];Create;True;0;0;0;False;0;False;1,0.9248579,0,0;0,0.8466554,2.411707,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;722;-4708.307,1257.708;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.LerpOp;555;1596.505,1674.833;Inherit;False;3;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0,0,0,0;False;2;FLOAT;1;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;220;-905.2066,2766.617;Inherit;True;SNOW;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;664;-4623.624,-15.42196;Inherit;False;Property;_OreEmissionColor;Ore Emission Color;33;1;[HDR];Create;True;0;0;0;False;0;False;1,0.9248579,0,0;7.999999,1.335938,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.VertexColorNode;662;-4597.276,138.6388;Inherit;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;708;-4546.92,-184.8685;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.GetLocalVarNode;686;-4400.997,1608.522;Inherit;False;497;DECALSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;544;1848.151,1832.851;Inherit;False;220;SNOW;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.OneMinusNode;663;-4419.246,129.6459;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;681;-4453.659,1257.423;Inherit;True;2;2;0;FLOAT3;0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StaticSwitch;557;1660.104,1502.07;Inherit;False;Property;_TOPPROJECTIONONOFF;TOP PROJECTION ON/OFF;27;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;673;-4373.328,-126.8075;Inherit;True;2;2;0;FLOAT3;0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;683;-4172.178,1259.247;Inherit;True;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.VertexColorNode;609;2262.744,1867.103;Inherit;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleAddOpNode;546;2078.315,1654.505;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;676;-4122.16,-135.8221;Inherit;True;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.OneMinusNode;611;2447.844,1876.96;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;677;-3848.508,-165.1693;Inherit;False;Property;_OREEMISSIONONOFF;ORE EMISSION ON/OFF;32;0;Create;True;0;0;0;False;0;False;0;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;605;2300.437,1677.604;Inherit;False;Property;_OreColor;Ore Color;31;1;[HDR];Create;True;0;0;0;False;0;False;1,0.9248579,0,0;2,0.4804064,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.StaticSwitch;545;2221.333,1505.783;Inherit;False;Property;_SNOWONOFF;SNOW ON/OFF;19;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;607;2603.098,1509.857;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;685;-3618.371,1233.073;Inherit;False;Decalemission;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;684;-3592.922,-160.4622;Inherit;False;oreemission;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;691;632.5616,3126.088;Inherit;False;684;oreemission;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;617;180.4198,2899.334;Inherit;False;Constant;_Color0;Color 0;21;0;Create;True;0;0;0;False;0;False;1,1,1,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.GetLocalVarNode;689;625.5616,2992.088;Inherit;False;685;Decalemission;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;539;2825.295,1504.856;Inherit;True;COLOR;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;731;129.6662,2781.02;Inherit;False;Property;_Smoothness;Smoothness;1;0;Create;True;0;0;0;False;0;False;0.2;0.321;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;614;435.1118,2845.14;Inherit;False;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;412;622.3464,2629.384;Inherit;True;539;COLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleAddOpNode;690;919.5616,2986.088;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;0;1151.042,2720.431;Float;False;True;-1;3;;0;0;Standard;Polytope Studio/PT_Rock_Shader;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;False;False;False;False;False;Off;0;False;;0;False;;False;0;False;;0;False;;False;0;Opaque;0.5;True;True;0;False;Opaque;;Geometry;All;12;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;False;2;15;10;25;False;0.5;True;0;0;False;;0;False;;0;0;False;;0;False;;0;False;;0;False;;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Relative;0;;-1;-1;-1;-1;0;False;0;0;False;;-1;0;False;;0;0;0;False;0.1;False;;0;False;;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
Node;AmplifyShaderEditor.ToggleSwitchNode;219;-1268.595,2760.266;Inherit;True;Property;_SnowNoiseOnOff;Snow Noise On/Off;23;0;Create;True;0;0;0;False;0;False;1;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;637;161.9385,1683.596;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;558;1330.633,1772.831;Inherit;False;528;TOPPROJECTIONMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.PowerNode;582;-2065.896,5680.709;Inherit;True;True;2;0;FLOAT;0;False;1;FLOAT;5;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;738;-2111.369,5876.133;Inherit;False;Constant;_Float10;Float 10;36;0;Create;True;0;0;0;False;0;False;5;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;523;-2735.743,5945.399;Inherit;False;Property;_TopProjectionTextureCoverage;Top Projection Texture  Coverage;30;0;Create;True;0;0;0;False;0;False;1;0.427;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.DotProductOpNode;524;-2638.518,5694.458;Inherit;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,1,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.Vector3Node;519;-2876.606,5796.51;Inherit;False;Property;_Vector1;Vector 1;26;1;[HideInInspector];Create;True;0;0;0;False;0;False;0,1,0;0.1,1,0.1;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.ClampOpNode;584;-1781.775,5684.8;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;526;-2395.617,5689.701;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;581;-2179.521,5688.574;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;3;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;737;-2432.369,5959.133;Inherit;False;Constant;_Float9;Float 9;36;0;Create;True;0;0;0;False;0;False;3;0;0;3;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;522;-2874.917,5494.738;Inherit;False;Property;_TopProjectionTextureTiling;Top Projection Texture Tiling;29;0;Create;True;0;0;0;False;0;False;0.5;1.46;0.1;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;521;-3041.719,5176.221;Inherit;True;Property;_TopProjectionTexture;Top Projection Texture;28;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.WorldNormalVector;518;-2860.619,5628.822;Inherit;False;False;1;0;FLOAT3;0,0,1;False;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.WorldNormalVector;201;-2637.648,3117.292;Inherit;True;False;1;0;FLOAT3;0,0,1;False;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.ToggleSwitchNode;688;-3924.81,1230.908;Inherit;False;Property;_DECALEMISSIONONOFF;DECAL EMISSION ON/OFF;12;0;Create;True;0;0;0;False;0;False;1;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
WireConnection;641;0;640;2
WireConnection;641;1;620;2
WireConnection;736;0;641;0
WireConnection;736;1;735;0
WireConnection;623;0;736;0
WireConnection;623;1;621;0
WireConnection;625;0;623;0
WireConnection;625;1;624;0
WireConnection;490;0;491;0
WireConnection;627;0;625;0
WireConnection;629;0;628;0
WireConnection;629;1;626;0
WireConnection;629;2;627;0
WireConnection;498;0;490;0
WireConnection;630;0;629;0
WireConnection;643;0;541;0
WireConnection;643;1;644;0
WireConnection;173;0;175;0
WireConnection;173;1;175;0
WireConnection;173;2;175;0
WireConnection;173;9;273;0
WireConnection;173;3;183;0
WireConnection;210;0;202;0
WireConnection;200;0;199;0
WireConnection;414;0;173;0
WireConnection;211;0;201;0
WireConnection;211;1;204;0
WireConnection;209;0;200;1
WireConnection;209;1;200;3
WireConnection;495;0;496;0
WireConnection;208;0;203;0
WireConnection;634;1;541;0
WireConnection;634;0;637;0
WireConnection;554;0;210;0
WireConnection;554;1;211;0
WireConnection;215;0;208;0
WireConnection;215;1;205;0
WireConnection;213;0;209;0
WireConnection;213;1;207;0
WireConnection;730;0;729;0
WireConnection;497;0;495;4
WireConnection;699;0;698;4
WireConnection;699;1;700;0
WireConnection;699;2;727;0
WireConnection;715;0;714;4
WireConnection;715;1;713;0
WireConnection;715;2;726;0
WireConnection;552;0;554;0
WireConnection;552;2;553;0
WireConnection;612;0;542;0
WireConnection;612;1;634;0
WireConnection;363;0;215;0
WireConnection;363;1;552;0
WireConnection;525;0;521;0
WireConnection;525;9;520;0
WireConnection;525;3;522;0
WireConnection;703;0;701;0
WireConnection;703;1;702;0
WireConnection;703;2;699;0
WireConnection;718;0;717;0
WireConnection;718;1;716;0
WireConnection;718;2;715;0
WireConnection;543;1;634;0
WireConnection;543;0;612;0
WireConnection;216;0;213;0
WireConnection;216;1;214;0
WireConnection;218;0;216;0
WireConnection;218;1;363;0
WireConnection;720;0;718;0
WireConnection;720;1;719;0
WireConnection;596;0;543;0
WireConnection;596;1;597;0
WireConnection;596;2;598;0
WireConnection;704;0;703;0
WireConnection;704;1;705;0
WireConnection;527;0;525;0
WireConnection;528;0;584;0
WireConnection;707;0;723;0
WireConnection;707;1;704;0
WireConnection;721;0;728;0
WireConnection;721;1;720;0
WireConnection;600;1;543;0
WireConnection;600;0;596;0
WireConnection;722;0;679;0
WireConnection;722;1;721;0
WireConnection;555;0;600;0
WireConnection;555;1;556;0
WireConnection;555;2;558;0
WireConnection;220;0;219;0
WireConnection;708;0;659;0
WireConnection;708;1;707;0
WireConnection;663;0;662;4
WireConnection;681;0;722;0
WireConnection;681;1;680;0
WireConnection;557;1;600;0
WireConnection;557;0;555;0
WireConnection;673;0;708;0
WireConnection;673;1;664;0
WireConnection;683;0;681;0
WireConnection;683;1;686;0
WireConnection;546;0;557;0
WireConnection;546;1;544;0
WireConnection;676;0;673;0
WireConnection;676;1;663;0
WireConnection;611;0;609;4
WireConnection;677;1;676;0
WireConnection;545;1;557;0
WireConnection;545;0;546;0
WireConnection;607;0;545;0
WireConnection;607;1;605;0
WireConnection;607;2;611;0
WireConnection;685;0;688;0
WireConnection;684;0;677;0
WireConnection;539;0;607;0
WireConnection;614;0;731;0
WireConnection;614;1;617;0
WireConnection;690;0;689;0
WireConnection;690;1;691;0
WireConnection;0;0;412;0
WireConnection;0;2;690;0
WireConnection;0;4;614;0
WireConnection;219;0;363;0
WireConnection;219;1;218;0
WireConnection;637;0;632;0
WireConnection;637;1;643;0
WireConnection;582;0;581;0
WireConnection;582;1;738;0
WireConnection;524;0;518;0
WireConnection;524;1;519;0
WireConnection;584;0;582;0
WireConnection;526;0;524;0
WireConnection;526;1;523;0
WireConnection;581;0;526;0
WireConnection;581;1;737;0
WireConnection;688;1;683;0
ASEEND*/
//CHKSM=C14125782570BBD8D71458B4CE210B120E67E03B