// Made with Amplify Shader Editor v1.9.1.5
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/ PT_Medieval Buildings Shader PBR"
{
	Properties
	{
		[HDR][Header(WALLS )]_Exteriorwalls1colour("Exterior walls 1 colour", Color) = (0.6792453,0.6010633,0.5863296,1)
		[HDR]_Exteriorwalls2color("Exterior walls 2 color", Color) = (0.3524386,0.6133218,0.754717,1)
		[HDR]_Exteriorwalls3color("Exterior walls 3 color", Color) = (0.8867924,0.6561894,0.23843,1)
		[HDR]_Interiorwallscolor("Interior walls color", Color) = (0.4127358,0.490063,0.5,0)
		[Header(EXTERIOR WALLS  DETAILS)][Toggle(_EXTERIORTEXTUREONOFF_ON)] _ExteriortextureOnOff("Exterior texture On/Off", Float) = 0
		[NoScaleOffset]_Exteriorwallstexture("Exterior walls texture", 2D) = "white" {}
		_Exteriorwallstiling("Exterior walls tiling", Range( 0 , 1)) = 0
		[Header(INTERIOR WALLS  DETAILS)][Toggle(_INTERIORTEXTUREONOFF_ON)] _InteriortextureOnOff("Interior texture On/Off", Float) = 0
		[NoScaleOffset]_Interiorwallstexture("Interior walls texture", 2D) = "white" {}
		_Interiorwallstiling("Interior walls tiling", Range( 0 , 1)) = 0
		[HDR][Header(WOOD)]_Wood1color("Wood 1 color", Color) = (0.4056604,0.2683544,0.135858,1)
		[HDR]_Wood2color("Wood 2 color", Color) = (0.1981132,0.103908,0.06634924,1)
		[HDR]_Wood3color("Wood 3 color", Color) = (0.5377358,0.4531547,0.377937,1)
		[HDR][Header(FABRICS)]_Fabric1color("Fabric 1 color", Color) = (0.5849056,0.5418971,0.4331613,0)
		[HDR]_Fabric2color("Fabric 2 color", Color) = (0.3649431,0.5566038,0.4386422,0)
		[HDR]_Fabric3color("Fabric 3 color", Color) = (0.5450981,0.6936808,0.6980392,0)
		[HDR][Header(ROCK )]_Rock1color("Rock 1 color", Color) = (0.4127358,0.490063,0.5,0)
		[HDR]_Rock2color("Rock 2 color", Color) = (0.3679245,0.2968027,0.1787558,0)
		[HDR][Header(CERAMIC TILES)]_Ceramictiles1color("Ceramic tiles 1 color", Color) = (0.3207547,0.04869195,0.01059096,1)
		_Ceramic1smoothness("Ceramic 1 smoothness", Range( 0 , 1)) = 0.3
		[HDR]_Ceramictiles2color("Ceramic tiles 2 color", Color) = (0.7924528,0.3776169,0.1682093,1)
		_Ceramic2smoothness("Ceramic 2 smoothness", Range( 0 , 1)) = 0.3
		[HDR]_Ceramictiles3color("Ceramic tiles 3 color ", Color) = (0.4677838,0.3813261,0.2501584,1)
		_Ceramic3smoothness("Ceramic 3 smoothness", Range( 0 , 1)) = 0.3
		[HDR][Header(METAL)]_Metal1color("Metal 1 color", Color) = (0.385947,0.5532268,0.5566038,0)
		_Metal1metallic("Metal 1 metallic", Range( 0 , 1)) = 0.65
		_Metal1smootness("Metal 1 smootness", Range( 0 , 1)) = 0.7
		[HDR]_Metal2color("Metal 2 color", Color) = (2,0.5960785,0,0)
		_Metal2metallic("Metal 2 metallic", Range( 0 , 1)) = 0.65
		_Metal2smootness("Metal 2 smootness", Range( 0 , 1)) = 0.7
		[HDR]_Metal3color("Metal 3 color", Color) = (1.584906,0.8017758,0,1)
		_Metal3metallic("Metal 3 metallic", Range( 0 , 1)) = 0.65
		_Metal3smootness("Metal 3 smootness", Range( 0 , 1)) = 0.7
		[HDR][Header(OTHER COLORS)]_Ropecolor("Rope color", Color) = (0.6037736,0.5810711,0.3389106,1)
		[HDR]_Haycolor("Hay color", Color) = (0.764151,0.517899,0.1622019,1)
		[HDR]_Mortarcolor("Mortar color", Color) = (0.6415094,0.5745595,0.4629761,0)
		[HDR]_Coatofarmscolor("Coat of arms color", Color) = (1,0,0,0)
		[NoScaleOffset]_Coarofarmstexture("Coar of arms texture", 2D) = "black" {}
		[Toggle]_MetallicOn("Metallic On", Float) = 1
		[Toggle]_SmoothnessOn("Smoothness On", Float) = 1
		[HideInInspector][Gamma]_Transparency("Transparency", Range( 0 , 1)) = 1
		[HideInInspector]_TextureSample2("Texture Sample 2", 2D) = "white" {}
		[HideInInspector][NoScaleOffset]_TextureSample9("Texture Sample 9", 2D) = "white" {}
		[HideInInspector] _texcoord2( "", 2D ) = "white" {}
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "Opaque"  "Queue" = "Geometry+0" "IgnoreProjector" = "True" }
		Cull Off
		Blend SrcAlpha OneMinusSrcAlpha
		
		AlphaToMask On
		CGINCLUDE
		#include "UnityPBSLighting.cginc"
		#include "Lighting.cginc"
		#pragma target 3.5
		#pragma shader_feature_local _EXTERIORTEXTUREONOFF_ON
		#pragma shader_feature_local _INTERIORTEXTUREONOFF_ON
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
			float2 uv2_texcoord2;
			float3 worldPos;
			half3 worldNormal;
			INTERNAL_DATA
		};

		uniform sampler2D _TextureSample2;
		uniform half4 _TextureSample2_ST;
		uniform half4 _Interiorwallscolor;
		uniform sampler2D _TextureSample9;
		uniform half4 _Mortarcolor;
		uniform half4 _Rock1color;
		uniform half4 _Rock2color;
		uniform half4 _Fabric1color;
		uniform half4 _Fabric2color;
		uniform half4 _Fabric3color;
		uniform half4 _Exteriorwalls1colour;
		uniform half4 _Exteriorwalls2color;
		uniform half4 _Exteriorwalls3color;
		uniform half4 _Wood1color;
		uniform half4 _Wood2color;
		uniform half4 _Wood3color;
		uniform half4 _Ceramictiles1color;
		uniform half4 _Ceramictiles2color;
		uniform half4 _Ceramictiles3color;
		uniform half4 _Ropecolor;
		uniform half4 _Haycolor;
		uniform half4 _Metal1color;
		uniform half4 _Metal2color;
		uniform half4 _Metal3color;
		uniform half4 _Coatofarmscolor;
		uniform sampler2D _Coarofarmstexture;
		uniform sampler2D _Interiorwallstexture;
		uniform half _Interiorwallstiling;
		uniform sampler2D _Exteriorwallstexture;
		uniform half _Exteriorwallstiling;
		uniform half _MetallicOn;
		uniform half _Metal1metallic;
		uniform half _Metal2metallic;
		uniform half _Metal3metallic;
		uniform half _SmoothnessOn;
		uniform half _Ceramic1smoothness;
		uniform half _Ceramic2smoothness;
		uniform half _Ceramic3smoothness;
		uniform half _Metal1smootness;
		uniform half _Metal2smootness;
		uniform half _Metal3smootness;
		uniform float _Transparency;


		inline float4 TriplanarSampling322( sampler2D topTexMap, sampler2D midTexMap, sampler2D botTexMap, float3 worldPos, float3 worldNormal, float falloff, float2 tiling, float3 normalScale, float3 index )
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


		inline float4 TriplanarSampling298( sampler2D topTexMap, sampler2D midTexMap, sampler2D botTexMap, float3 worldPos, float3 worldNormal, float falloff, float2 tiling, float3 normalScale, float3 index )
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


		void surf( Input i , inout SurfaceOutputStandard o )
		{
			o.Normal = float3(0,0,1);
			float2 uv_TextureSample2 = i.uv_texcoord * _TextureSample2_ST.xy + _TextureSample2_ST.zw;
			half4 BASETEXTURE243 = tex2D( _TextureSample2, uv_TextureSample2 );
			half4 color315 = IsGammaSpace() ? half4(0.1607843,1,0,1) : half4(0.02217388,1,0,1);
			float2 uv_TextureSample9120 = i.uv_texcoord;
			half4 MASKTEXTURE251 = tex2D( _TextureSample9, uv_TextureSample9120 );
			half temp_output_310_0 = saturate( ( 1.0 - ( ( distance( color315.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult309 = lerp( float4( 0,0,0,0 ) , ( BASETEXTURE243 * _Interiorwallscolor ) , temp_output_310_0);
			half4 color314 = IsGammaSpace() ? half4(0.4392157,0,0.4392157,1) : half4(0.1620294,0,0.1620294,1);
			half4 lerpResult313 = lerp( lerpResult309 , ( BASETEXTURE243 * _Mortarcolor ) , saturate( ( 1.0 - ( ( distance( color314.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color130 = IsGammaSpace() ? half4(0,0.4784314,0.4784314,1) : half4(0,0.1946179,0.1946179,1);
			half4 lerpResult132 = lerp( lerpResult313 , ( BASETEXTURE243 * _Rock1color ) , saturate( ( 1.0 - ( ( distance( color130.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color134 = IsGammaSpace() ? half4(0,1,0.7294118,1) : half4(0,1,0.4910209,1);
			half4 lerpResult133 = lerp( lerpResult132 , ( BASETEXTURE243 * _Rock2color ) , saturate( ( 1.0 - ( ( distance( color134.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color121 = IsGammaSpace() ? half4(0.6196079,0.9333334,1,1) : half4(0.3419145,0.8549928,1,1);
			half4 lerpResult124 = lerp( lerpResult133 , ( BASETEXTURE243 * _Fabric1color ) , saturate( ( 1.0 - ( ( distance( color121.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color138 = IsGammaSpace() ? half4(0.9333334,1,0.6196079,1) : half4(0.8549928,1,0.3419145,1);
			half4 lerpResult126 = lerp( lerpResult124 , ( BASETEXTURE243 * _Fabric2color ) , saturate( ( 1.0 - ( ( distance( color138.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color136 = IsGammaSpace() ? half4(1,0.6196079,0.9333334,1) : half4(1,0.3419145,0.8549928,1);
			half4 lerpResult9 = lerp( lerpResult126 , ( BASETEXTURE243 * _Fabric3color ) , saturate( ( 1.0 - ( ( distance( color136.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color107 = IsGammaSpace() ? half4(1,1,0,1) : half4(1,1,0,1);
			half temp_output_7_0 = saturate( ( 1.0 - ( ( distance( color107.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult10 = lerp( lerpResult9 , ( BASETEXTURE243 * _Exteriorwalls1colour ) , temp_output_7_0);
			half4 color106 = IsGammaSpace() ? half4(1,0,1,1) : half4(1,0,1,1);
			half temp_output_11_0 = saturate( ( 1.0 - ( ( distance( color106.rgb , MASKTEXTURE251.rgb ) - 0.0 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult15 = lerp( lerpResult10 , ( BASETEXTURE243 * _Exteriorwalls2color ) , temp_output_11_0);
			half4 color105 = IsGammaSpace() ? half4(0,1,1,1) : half4(0,1,1,1);
			half temp_output_13_0 = saturate( ( 1.0 - ( ( distance( color105.rgb , MASKTEXTURE251.rgb ) - 0.0 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult19 = lerp( lerpResult15 , ( BASETEXTURE243 * _Exteriorwalls3color ) , temp_output_13_0);
			half4 color103 = IsGammaSpace() ? half4(0.6862745,0.8352942,0.8352942,1) : half4(0.4286906,0.6653875,0.6653875,1);
			half4 lerpResult22 = lerp( lerpResult19 , ( BASETEXTURE243 * _Wood1color ) , saturate( ( 1.0 - ( ( distance( color103.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color104 = IsGammaSpace() ? half4(1,0.7294118,0,1) : half4(1,0.4910209,0,1);
			half4 lerpResult27 = lerp( lerpResult22 , ( BASETEXTURE243 * _Wood2color ) , saturate( ( 1.0 - ( ( distance( color104.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color102 = IsGammaSpace() ? half4(0.7294118,0,1,1) : half4(0.4910209,0,1,1);
			half4 lerpResult32 = lerp( lerpResult27 , ( BASETEXTURE243 * _Wood3color ) , saturate( ( 1.0 - ( ( distance( color102.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color17 = IsGammaSpace() ? half4(0.3960785,0.2627451,0.02352941,1) : half4(0.1301365,0.05612849,0.001821162,1);
			half temp_output_92_0 = saturate( ( 1.0 - ( ( distance( color17.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult91 = lerp( lerpResult32 , ( BASETEXTURE243 * _Ceramictiles1color ) , temp_output_92_0);
			half4 color24 = IsGammaSpace() ? half4(0.5372549,0.4313726,0.2509804,1) : half4(0.2501584,0.1559265,0.05126947,1);
			half temp_output_95_0 = saturate( ( 1.0 - ( ( distance( color24.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult93 = lerp( lerpResult91 , ( BASETEXTURE243 * _Ceramictiles2color ) , temp_output_95_0);
			half4 color94 = IsGammaSpace() ? half4(0.7137255,0.6509804,0.5372549,1) : half4(0.4677838,0.3813261,0.2501584,1);
			half temp_output_96_0 = saturate( ( 1.0 - ( ( distance( color94.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult39 = lerp( lerpResult93 , ( BASETEXTURE243 * _Ceramictiles3color ) , temp_output_96_0);
			half4 color84 = IsGammaSpace() ? half4(0,0.1294118,0.5058824,1) : half4(0,0.01520852,0.2195262,1);
			half4 lerpResult71 = lerp( lerpResult39 , ( BASETEXTURE243 * _Ropecolor ) , saturate( ( 1.0 - ( ( distance( color84.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color85 = IsGammaSpace() ? half4(1,0.4313726,0.5254902,1) : half4(1,0.1559265,0.2383977,1);
			half4 lerpResult72 = lerp( lerpResult71 , ( BASETEXTURE243 * _Haycolor ) , saturate( ( 1.0 - ( ( distance( color85.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) ));
			half4 color38 = IsGammaSpace() ? half4(0.8274511,0.8784314,0.6980392,1) : half4(0.6514059,0.7454044,0.4452012,1);
			half temp_output_41_0 = saturate( ( 1.0 - ( ( distance( color38.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult47 = lerp( lerpResult72 , ( BASETEXTURE243 * _Metal1color ) , temp_output_41_0);
			half4 color117 = IsGammaSpace() ? half4(0.6392157,0.6784314,0.5411765,1) : half4(0.3662527,0.4178852,0.2541522,1);
			half temp_output_116_0 = saturate( ( 1.0 - ( ( distance( color117.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult150 = lerp( lerpResult47 , ( BASETEXTURE243 * _Metal2color ) , temp_output_116_0);
			half4 color118 = IsGammaSpace() ? half4(0.4627451,0.4901961,0.3921569,1) : half4(0.1811642,0.2050788,0.1274377,1);
			half temp_output_149_0 = saturate( ( 1.0 - ( ( distance( color118.rgb , MASKTEXTURE251.rgb ) - 0.1 ) / max( 0.0 , 1E-05 ) ) ) );
			half4 lerpResult151 = lerp( lerpResult150 , ( BASETEXTURE243 * _Metal3color ) , temp_output_149_0);
			float2 uv1_Coarofarmstexture157 = i.uv2_texcoord2;
			half temp_output_49_0 = ( 1.0 - tex2D( _Coarofarmstexture, uv1_Coarofarmstexture157 ).a );
			half4 temp_cast_42 = (temp_output_49_0).xxxx;
			half4 temp_output_1_0_g169 = temp_cast_42;
			half4 color54 = IsGammaSpace() ? half4(0,0,0,0) : half4(0,0,0,0);
			half4 temp_output_2_0_g169 = color54;
			half temp_output_11_0_g169 = distance( temp_output_1_0_g169 , temp_output_2_0_g169 );
			half2 _Vector0 = half2(1.6,1);
			half4 lerpResult21_g169 = lerp( _Coatofarmscolor , temp_output_1_0_g169 , saturate( ( ( temp_output_11_0_g169 - _Vector0.x ) / max( _Vector0.y , 1E-05 ) ) ));
			half temp_output_156_0 = ( 1.0 - temp_output_49_0 );
			half4 lerpResult165 = lerp( lerpResult151 , lerpResult21_g169 , temp_output_156_0);
			half2 temp_cast_43 = (_Interiorwallstiling).xx;
			float3 ase_worldPos = i.worldPos;
			half3 ase_worldNormal = WorldNormalVector( i, half3( 0, 0, 1 ) );
			float4 triplanar322 = TriplanarSampling322( _Interiorwallstexture, _Interiorwallstexture, _Interiorwallstexture, ase_worldPos, ase_worldNormal, 1.0, temp_cast_43, float3( 1,1,1 ), float3(0,0,0) );
			float4 INDETAILTEXTUREvar323 = triplanar322;
			half4 blendOpSrc325 = INDETAILTEXTUREvar323;
			half4 blendOpDest325 = lerpResult165;
			half INTWALLSMASK329 = temp_output_310_0;
			half4 lerpBlendMode325 = lerp(blendOpDest325,( blendOpSrc325 * blendOpDest325 ),INTWALLSMASK329);
			#ifdef _INTERIORTEXTUREONOFF_ON
				half4 staticSwitch328 = ( saturate( lerpBlendMode325 ));
			#else
				half4 staticSwitch328 = lerpResult165;
			#endif
			half2 temp_cast_45 = ((0.1 + (_Exteriorwallstiling - 0.0) * (0.4 - 0.1) / (1.0 - 0.0))).xx;
			float4 triplanar298 = TriplanarSampling298( _Exteriorwallstexture, _Exteriorwallstexture, _Exteriorwallstexture, ase_worldPos, ase_worldNormal, 10.0, temp_cast_45, float3( 1,1,1 ), float3(0,0,0) );
			half4 OUTDETAILTEXTUREvar299 = triplanar298;
			half4 blendOpSrc231 = OUTDETAILTEXTUREvar299;
			half4 blendOpDest231 = staticSwitch328;
			half WALLSMASK227 = ( temp_output_7_0 + temp_output_11_0 + temp_output_13_0 );
			half4 lerpBlendMode231 = lerp(blendOpDest231,( blendOpSrc231 * blendOpDest231 ),WALLSMASK227);
			#ifdef _EXTERIORTEXTUREONOFF_ON
				half4 staticSwitch266 = ( saturate( lerpBlendMode231 ));
			#else
				half4 staticSwitch266 = staticSwitch328;
			#endif
			o.Albedo = staticSwitch266.rgb;
			half lerpResult110 = lerp( 0.0 , _Metal1metallic , temp_output_41_0);
			half lerpResult112 = lerp( lerpResult110 , _Metal2metallic , temp_output_116_0);
			half lerpResult113 = lerp( lerpResult112 , _Metal3metallic , temp_output_149_0);
			half lerpResult55 = lerp( lerpResult113 , 0.0 , temp_output_156_0);
			o.Metallic = (( _MetallicOn )?( lerpResult55 ):( 0.0 ));
			half lerpResult26 = lerp( 0.0 , _Ceramic1smoothness , temp_output_92_0);
			half lerpResult31 = lerp( lerpResult26 , _Ceramic2smoothness , temp_output_95_0);
			half lerpResult34 = lerp( lerpResult31 , _Ceramic3smoothness , temp_output_96_0);
			half lerpResult42 = lerp( lerpResult34 , _Metal1smootness , temp_output_41_0);
			half lerpResult43 = lerp( lerpResult42 , _Metal2smootness , temp_output_116_0);
			half lerpResult46 = lerp( lerpResult43 , _Metal3smootness , 0.0);
			o.Smoothness = (( _SmoothnessOn )?( lerpResult46 ):( 0.0 ));
			o.Alpha = _Transparency;
		}

		ENDCG
		CGPROGRAM
		#pragma surface surf Standard keepalpha fullforwardshadows 

		ENDCG
		Pass
		{
			Name "ShadowCaster"
			Tags{ "LightMode" = "ShadowCaster" }
			ZWrite On
			AlphaToMask Off
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
			sampler3D _DitherMaskLOD;
			struct v2f
			{
				V2F_SHADOW_CASTER;
				float4 customPack1 : TEXCOORD1;
				float4 tSpace0 : TEXCOORD2;
				float4 tSpace1 : TEXCOORD3;
				float4 tSpace2 : TEXCOORD4;
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
				SurfaceOutputStandard o;
				UNITY_INITIALIZE_OUTPUT( SurfaceOutputStandard, o )
				surf( surfIN, o );
				#if defined( CAN_SKIP_VPOS )
				float2 vpos = IN.pos;
				#endif
				half alphaRef = tex3D( _DitherMaskLOD, float3( vpos.xy * 0.25, o.Alpha * 0.9375 ) ).a;
				clip( alphaRef - 0.01 );
				SHADOW_CASTER_FRAGMENT( IN )
			}
			ENDCG
		}
	}
	Fallback "Diffuse"
	CustomEditor "ASEMaterialInspector"
}
/*ASEBEGIN
Version=19105
Node;AmplifyShaderEditor.CommentaryNode;2;-11727.8,601.6224;Inherit;False;2305.307;694.8573;Comment;15;138;137;136;135;128;127;126;125;124;123;122;121;66;9;6;FABRIC COLORS;0.05562881,0.9716981,0,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;3;-5798.301,-1438.134;Inherit;False;427.9199;359.978;Comment;1;120;MASK TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;4;-5653.832,-1956.559;Inherit;False;341.4902;248.4146;Comment;1;119;BASE TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;5;-9023.379,608.0663;Inherit;False;4637.659;753.5428;Comment;30;107;106;105;104;103;102;101;100;99;98;97;68;67;32;29;27;25;23;22;20;19;18;16;15;13;12;11;10;8;7;WALL&WOOD COLORS;0.735849,0.7152051,0.3158597,1;0;0
Node;AmplifyShaderEditor.FunctionNode;7;-8447.35,1034.486;Inherit;True;Color Mask;-1;;117;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;9;-9606.48,866.7899;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;10;-8102.76,876.0576;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;11;-7853.699,1034.76;Inherit;True;Color Mask;-1;;118;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;13;-7205.061,1056.685;Inherit;True;Color Mask;-1;;119;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;14;-4153.02,596.6636;Inherit;False;2359.399;695.7338;Comment;15;96;95;94;93;92;91;90;89;88;39;35;33;30;24;17;CERAMIC COLORS;0.4690726,0.7830189,0.47128,1;0;0
Node;AmplifyShaderEditor.LerpOp;15;-7478.76,860.0576;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;17;-3840.87,1096.28;Inherit;False;Constant;_Color10;Color 10;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.3960785,0.2627451,0.02352941,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;19;-6911.49,858.3113;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;20;-6449.77,1032.8;Inherit;True;Color Mask;-1;;121;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;21;-3597.649,2159.435;Inherit;False;1768.502;211.4459;Comment;6;77;76;75;34;31;26;CERAMIC  SMOOTHNESS;1,1,1,1;0;0
Node;AmplifyShaderEditor.LerpOp;22;-6208.12,859.874;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;24;-3178.109,1065.361;Inherit;False;Constant;_Color5;Color 5;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.5372549,0.4313726,0.2509804,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;25;-5872.45,1039.417;Inherit;True;Color Mask;-1;;139;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;26;-3248.52,2213.889;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;27;-5580.63,861.3362;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;28;-1645.72,576.5326;Inherit;False;2342.301;700.673;Comment;10;154;87;86;85;84;73;72;71;70;69;ROPE HAY  COLORS;0.7735849,0.5371538,0.1788003,1;0;0
Node;AmplifyShaderEditor.LerpOp;31;-2622.24,2214.881;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;32;-5008.12,842.1277;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;33;-2737.8,649.0425;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;34;-2013.149,2209.435;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;36;908.123,575.767;Inherit;False;2552.517;745.6387;Comment;15;153;152;151;150;149;148;147;118;117;116;111;74;47;41;38;METAL COLORS;0.259434,0.8569208,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;37;1257.24,2110.681;Inherit;False;2210.534;259.0801;Comment;6;81;80;79;46;43;42;METAL SMOOTHNESS;1,1,1,1;0;0
Node;AmplifyShaderEditor.LerpOp;39;-1977.62,861.8309;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;40;1246.527,1532.62;Inherit;False;2183.689;260.3257;Comment;6;115;114;113;112;110;78;METAL METALLIC;1,1,1,1;0;0
Node;AmplifyShaderEditor.FunctionNode;41;1555.567,1030.149;Inherit;True;Color Mask;-1;;146;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;42;1865.915,2207.542;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;43;2537.24,2190.681;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;44;3391.176,-994.6529;Inherit;False;1262.249;589.4722;;7;157;156;56;54;53;51;49;COAT OF ARMS;1,0,0.7651567,1;0;0
Node;AmplifyShaderEditor.LerpOp;46;3177.24,2190.681;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;47;1952.685,840.8496;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;69;-865.71,631.7888;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;70;-247.5977,628.9116;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;71;-684.6436,846.9166;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;72;-64.19238,844.6108;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;84;-1213.859,1111.218;Inherit;False;Constant;_Color13;Color 13;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0,0.1294118,0.5058824,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;85;-603.5586,1092.531;Inherit;False;Constant;_Color15;Color 15;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;1,0.4313726,0.5254902,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;86;-341.6494,1028.773;Inherit;True;Color Mask;-1;;155;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;87;-941.1699,1032.642;Inherit;True;Color Mask;-1;;156;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;91;-3174.84,867.0479;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;92;-3540.89,1032.595;Inherit;True;Color Mask;-1;;157;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;93;-2554.39,852.516;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;94;-2579.38,1083.599;Inherit;False;Constant;_Color11;Color 11;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.7137255,0.6509804,0.5372549,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;95;-2907.2,1036.439;Inherit;True;Color Mask;-1;;158;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;96;-2309.47,1036.677;Inherit;True;Color Mask;-1;;159;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;101;-5280.79,1033.502;Inherit;True;Color Mask;-1;;160;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;106;-8111.859,1153.812;Inherit;False;Constant;_Color12;Color 12;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;1,0,1,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;107;-8700.35,1150.605;Inherit;False;Constant;_Color6;Color 6;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;1,1,0,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;110;1862.643,1623.919;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;112;2582.651,1573.047;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;113;3165.043,1573.725;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FunctionNode;116;2266.754,970.8533;Inherit;True;Color Mask;-1;;162;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;117;2001.787,1117.835;Inherit;False;Constant;_Color7;Color 7;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.6392157,0.6784314,0.5411765,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;118;2676.376,1137.907;Inherit;False;Constant;_Color8;Color 7;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.4627451,0.4901961,0.3921569,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;121;-11434.19,1151.006;Inherit;False;Constant;_Color23;Color 23;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.6196079,0.9333334,1,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;122;-11173.81,1057.373;Inherit;True;Color Mask;-1;;163;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;124;-10803.72,872.0068;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;126;-10183.26,857.4746;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;129;-13256.22,655.2789;Inherit;False;1320.625;643.1454;Comment;10;141;134;133;140;143;130;131;132;139;142;ROCK COLORS;0.05562881,0.9716981,0,1;0;0
Node;AmplifyShaderEditor.FunctionNode;135;-9927.891,1039.456;Inherit;True;Color Mask;-1;;165;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;136;-10207.81,1113.681;Inherit;False;Constant;_Color25;Color 25;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;1,0.6196079,0.9333334,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;137;-10546.03,1020.571;Inherit;True;Color Mask;-1;;166;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;138;-10797.31,1094.721;Inherit;False;Constant;_Color24;Color 24;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.9333334,1,0.6196079,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;148;2382.021,705.0095;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;149;3049.118,978.9347;Inherit;True;Color Mask;-1;;168;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;150;2617.332,846.6812;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;151;3405.815,868.7943;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;102;-5535.17,1135.14;Inherit;False;Constant;_Color32;Color 32;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.7294118,0,1,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;103;-6723.689,1145.791;Inherit;False;Constant;_Color34;Color 34;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.6862745,0.8352942,0.8352942,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;104;-6118.91,1151.033;Inherit;False;Constant;_Color33;Color 33;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;1,0.7294118,0,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;105;-7474.15,1153.183;Inherit;False;Constant;_Color31;Color 31;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0,1,1,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.OneMinusNode;49;3768.177,-815.4178;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.Vector2Node;53;4160.477,-596.9551;Inherit;False;Constant;_Vector0;Vector 0;1;0;Create;True;0;0;0;False;0;False;1.6,1;0,0;0;3;FLOAT2;0;FLOAT;1;FLOAT;2
Node;AmplifyShaderEditor.ColorNode;54;4136.386,-959.0538;Inherit;False;Constant;_Color4;Color 4;1;0;Create;True;0;0;0;False;0;False;0,0,0,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;56;4389.274,-821.8439;Inherit;False;Replace Color;-1;;169;896dccb3016c847439def376a728b869;1,12,0;5;1;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;FLOAT;0;False;5;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.WireNode;59;4875.945,-795.9808;Inherit;False;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.WireNode;60;4817.969,-492.5753;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WireNode;62;4846.844,-440.6251;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WireNode;63;4907.966,-761.865;Inherit;False;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.OneMinusNode;156;3960.033,-541.5626;Inherit;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;55;4321.82,1564.807;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;123;-10928.77,702.3899;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;125;-10307.44,708.9993;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;6;-9785.64,681.6268;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;244;-10625.49,459.1433;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;245;-7923.934,434.5842;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;8;-8159.558,716.8089;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;12;-7647.213,709.4507;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;16;-7100.646,696.2062;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;18;-6349.994,716.7143;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;23;-5722.873,706.413;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;29;-5186.606,713.7711;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;246;-5924.16,459.5471;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;247;-2714.35,391.4492;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;30;-3308.171,735.1832;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;35;-2211.73,654.1045;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;248;-666.146,233.4418;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;147;1814.686,663.9406;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;153;3166.919,727.6151;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;253;-10471.74,1493.962;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleAddOpNode;226;-7558.565,1645.21;Inherit;True;3;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;294;-1265.791,-2023.716;Inherit;False;1516.12;1028.51;Comment;7;299;298;296;295;331;333;334;EX DETAIL TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;227;-6876.959,1632.748;Inherit;True;WALLSMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;254;-7947.586,1485.719;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;302;-6121.233,1509.821;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;303;-3062.826,1511.723;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;304;-710.8005,1590.452;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;305;2304.995,1430.737;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SamplerNode;119;-5623.402,-1895.385;Inherit;True;Property;_TextureSample2;Texture Sample 2;42;1;[HideInInspector];Create;True;0;0;0;False;0;False;-1;34892be16c52f1948b4b5666052d6bf5;34892be16c52f1948b4b5666052d6bf5;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RegisterLocalVarNode;243;-5237.781,-1904.64;Inherit;False;BASETEXTURE;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;251;-5227.111,-1537.049;Inherit;False;MASKTEXTURE;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;249;2192.794,383.184;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;139;-12802.18,739.1353;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;132;-12636.8,893.1993;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;131;-12912,1013.638;Inherit;False;Color Mask;-1;;170;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;140;-12182.57,703.0303;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;133;-12146.87,899.1673;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;134;-12503.74,1103.401;Inherit;False;Constant;_Color27;Color 24;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0,1,0.7294118,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;130;-13129.93,1069.544;Inherit;False;Constant;_Color26;Color 23;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0,0.4784314,0.4784314,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;141;-12298.32,1018.326;Inherit;False;Color Mask;-1;;171;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;252;-12793.74,1330.962;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;250;-12717.54,538.4932;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;306;-13304.54,1906.265;Inherit;False;1320.625;643.1454;Comment;10;316;315;314;313;312;311;310;309;308;307;INTERIOR&MORTAR;0.05562881,0.9716981,0,1;0;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;308;-12850.5,1990.122;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;309;-12685.12,2144.185;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.FunctionNode;310;-12960.32,2264.624;Inherit;True;Color Mask;-1;;172;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;312;-12230.89,1954.016;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;313;-12195.19,2150.154;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;314;-12552.06,2354.387;Inherit;False;Constant;_Color28;Color 24;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.4392157,0,0.4392157,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;315;-13178.25,2320.53;Inherit;False;Constant;_Color29;Color 23;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.1607843,1,0,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;316;-12346.64,2269.312;Inherit;True;Color Mask;-1;;173;eec747d987850564c95bde0e5a6d1867;0;4;1;FLOAT3;0,0,0;False;3;FLOAT3;0,0,0;False;4;FLOAT;0.1;False;5;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;318;-12740.91,1783.242;Inherit;False;243;BASETEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SamplerNode;120;-5741.702,-1349.504;Inherit;True;Property;_TextureSample9;Texture Sample 9;43;2;[HideInInspector];[NoScaleOffset];Create;True;0;0;0;False;0;False;120;c91055552562c3941a1c318e8d5bc5c5;c91055552562c3941a1c318e8d5bc5c5;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.CommentaryNode;319;866.049,-2029.98;Inherit;False;1516.12;1028.51;Comment;5;323;322;320;330;332;IN DETAIL TEXTURE;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;299;-223.4873,-1739.599;Inherit;True;OUTDETAILTEXTUREvar;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;329;-12363.67,2799.661;Inherit;False;INTWALLSMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldPosInputsNode;320;905.0309,-1581.239;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.GetLocalVarNode;317;-12817.11,2575.712;Inherit;False;251;MASKTEXTURE;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;143;-12424.21,700.1229;Inherit;False;Property;_Rock2color;Rock 2 color;18;1;[HDR];Create;True;0;0;0;False;0;False;0.3679245,0.2968027,0.1787558,0;0.2023368,0,0.4339623,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;332;1068.05,-1383.311;Inherit;False;Property;_Interiorwallstiling;Interior walls tiling;10;0;Create;True;0;0;0;False;0;False;0;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;51;4140.196,-779.3428;Inherit;False;Property;_Coatofarmscolor;Coat of arms color;37;1;[HDR];Create;True;0;0;0;False;0;False;1,0,0,0;1,0.0990566,0.0990566,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;128;-10596.59,675.0156;Inherit;False;Property;_Fabric2color;Fabric 2 color;15;1;[HDR];Create;True;0;0;0;False;0;False;0.3649431,0.5566038,0.4386422,0;0.2023368,0,0.4339623,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;66;-10024.01,673.8386;Inherit;False;Property;_Fabric3color;Fabric 3 color;16;1;[HDR];Create;True;0;0;0;False;0;False;0.5450981,0.6936808,0.6980392,0;0.3773585,0,0.06650025,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;98;-6027.79,655.4275;Inherit;False;Property;_Wood2color;Wood 2 color;12;1;[HDR];Create;True;0;0;0;False;0;False;0.1981132,0.103908,0.06634924,1;0.6792453,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;99;-5511.38,669.605;Inherit;False;Property;_Wood3color;Wood 3 color;13;1;[HDR];Create;True;0;0;0;False;0;False;0.5377358,0.4531547,0.377937,1;0.7735849,0.492613,0.492613,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;89;-3013.319,661.918;Inherit;False;Property;_Ceramictiles2color;Ceramic tiles 2 color;21;1;[HDR];Create;True;0;0;0;False;0;False;0.7924528,0.3776169,0.1682093,1;1,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;90;-2432.66,671.0714;Inherit;False;Property;_Ceramictiles3color;Ceramic tiles 3 color ;23;1;[HDR];Create;True;0;0;0;False;0;False;0.4677838,0.3813261,0.2501584,1;0,0.1142961,0.1698113,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;111;2121.972,708.9114;Inherit;False;Property;_Metal2color;Metal 2 color;28;1;[HDR];Create;True;0;0;0;False;0;False;2,0.5960785,0,0;0.9528302,0.9528302,0.9528302,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;152;2826.721,692.4199;Inherit;False;Property;_Metal3color;Metal 3 color;31;1;[HDR];Create;True;0;0;0;False;0;False;1.584906,0.8017758,0,1;0.3301887,0.3301887,0.3301887,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;38;1330.412,1127.181;Inherit;False;Constant;_Color17;Color 17;51;1;[HideInInspector];Create;True;0;0;0;False;0;False;0.8274511,0.8784314,0.6980392,1;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;78;1558.641,1655.919;Inherit;False;Property;_Metal1metallic;Metal 1 metallic;26;0;Create;True;0;0;0;False;0;False;0.65;0.903;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;114;2229.969,1639.058;Inherit;False;Property;_Metal2metallic;Metal 2 metallic;29;0;Create;True;0;0;0;False;0;False;0.65;0.903;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;115;2864.037,1633.553;Inherit;False;Property;_Metal3metallic;Metal 3 metallic;32;0;Create;True;0;0;0;False;0;False;0.65;0.903;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;79;1535.77,2263.888;Inherit;False;Property;_Metal1smootness;Metal 1 smootness;27;0;Create;True;0;0;0;False;0;False;0.7;0.721;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;80;2247.969,2207.952;Inherit;False;Property;_Metal2smootness;Metal 2 smootness;30;0;Create;True;0;0;0;False;0;False;0.7;0.721;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;81;2873.24,2206.681;Inherit;False;Property;_Metal3smootness;Metal 3 smootness;33;0;Create;True;0;0;0;False;0;False;0.7;0.7;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;75;-3547.649,2236.753;Inherit;False;Property;_Ceramic1smoothness;Ceramic 1 smoothness;20;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;76;-2915.24,2236.934;Inherit;False;Property;_Ceramic2smoothness;Ceramic 2 smoothness;22;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;77;-2332.96,2231.924;Inherit;False;Property;_Ceramic3smoothness;Ceramic 3 smoothness;24;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;73;-540.9141,659.7532;Inherit;False;Property;_Haycolor;Hay color;35;2;[HDR];[Header];Create;True;0;0;0;False;0;False;0.764151,0.517899,0.1622019,1;0.4245283,0.190437,0.09011215,1;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SamplerNode;157;3415.809,-925.424;Inherit;True;Property;_Coarofarmstexture;Coar of arms texture;38;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;-1;None;None;True;1;False;black;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;100;-7926.76,668.0576;Inherit;False;Property;_Exteriorwalls2color;Exterior walls 2 color;2;1;[HDR];Create;True;0;0;0;False;0;False;0.3524386,0.6133218,0.754717,1;0.6792453,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;68;-7382.76,670.9277;Inherit;False;Property;_Exteriorwalls3color;Exterior walls 3 color;3;1;[HDR];Create;True;0;0;0;False;0;False;0.8867924,0.6561894,0.23843,1;0.7735849,0.492613,0.492613,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TriplanarNode;322;1437.313,-1750.768;Inherit;True;Cylindrical;World;False;Top Texture 0;_TopTexture0;white;3;None;Mid Texture 0;_MidTexture0;white;6;None;Bot Texture 0;_BotTexture0;white;5;None;Triplanar Sampler;Tangent;10;0;SAMPLER2D;;False;5;FLOAT;1;False;1;SAMPLER2D;;False;6;FLOAT;0;False;2;SAMPLER2D;;False;7;FLOAT;0;False;9;FLOAT3;0,0,0;False;8;FLOAT3;1,1,1;False;3;FLOAT2;1,1;False;4;FLOAT;1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TexturePropertyNode;330;896.6069,-1789.026;Inherit;True;Property;_Interiorwallstexture;Interior walls texture;9;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.RegisterLocalVarNode;323;1972.037,-1755.476;Float;True;INDETAILTEXTUREvar;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.ColorNode;311;-12469.53,1950.109;Inherit;False;Property;_Mortarcolor;Mortar color;36;1;[HDR];Create;True;0;0;0;False;0;False;0.6415094,0.5745595,0.4629761,0;0.2023368,0,0.4339623,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;0;7961.389,723.4713;Half;False;True;-1;3;ASEMaterialInspector;0;0;Standard;Polytope Studio/ PT_Medieval Buildings Shader PBR;False;False;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;False;False;False;Off;0;False;;0;False;;False;1;False;;1;False;;False;2;Custom;0.5;True;True;0;True;Opaque;;Geometry;All;12;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;False;2;15;10;25;False;0.5;True;2;5;False;;10;False;;0;5;False;;1;False;;0;False;;0;False;;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Relative;0;;0;-1;-1;-1;0;True;0;0;False;;-1;0;False;_Transparency;0;0;0;False;0;False;;0;False;_Transparency;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
Node;AmplifyShaderEditor.RangedFloatNode;171;7357.75,2121.079;Float;False;Property;_Transparency;Transparency;41;2;[HideInInspector];[Gamma];Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.WireNode;64;4840.567,823.2047;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WireNode;65;4887.951,785.4938;Inherit;False;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;165;5056.114,785.393;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.BlendOpsNode;325;6026.704,822.2209;Inherit;False;Multiply;True;3;0;FLOAT4;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;326;5770.72,998.0026;Inherit;True;329;INTWALLSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;327;5758.699,679.7779;Inherit;False;323;INDETAILTEXTUREvar;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;223;6908.065,611.3265;Inherit;False;299;OUTDETAILTEXTUREvar;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.WorldPosInputsNode;295;-1369.173,-1657.575;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.TFHCRemapNode;334;-846.7884,-1352.031;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0.1;False;4;FLOAT;0.4;False;1;FLOAT;0
Node;AmplifyShaderEditor.BlendOpsNode;231;7249.958,752.8592;Inherit;False;Multiply;True;3;0;FLOAT4;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;228;6929.926,865.3324;Inherit;True;227;WALLSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;166;7032.947,1537.321;Inherit;True;Property;_MetallicOn;Metallic On;39;0;Create;True;0;0;0;False;0;False;1;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;173;7046.772,2121.252;Inherit;True;Property;_SmoothnessOn;Smoothness On;40;0;Create;True;0;0;0;False;0;False;1;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;50;4022.734,1640.753;Inherit;False;Constant;_Float1;Float 0;50;0;Create;True;0;0;0;False;0;False;0;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;88;-3641.46,664.2064;Inherit;False;Property;_Ceramictiles1color;Ceramic tiles 1 color;19;2;[HDR];[Header];Create;True;1;CERAMIC TILES;0;0;False;0;False;0.3207547,0.04869195,0.01059096,1;0.9056604,0.6815338,0.4229263,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;142;-13099.11,707.9922;Inherit;False;Property;_Rock1color;Rock 1 color;17;2;[HDR];[Header];Create;True;1;ROCK ;0;0;False;0;False;0.4127358,0.490063,0.5,0;0,0.1132075,0.01206957,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;307;-13140.54,1954.301;Inherit;False;Property;_Interiorwallscolor;Interior walls color;4;1;[HDR];Create;True;0;0;0;False;0;False;0.4127358,0.490063,0.5,0;0,0.1132075,0.01206957,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;127;-11196.36,713.4846;Inherit;False;Property;_Fabric1color;Fabric 1 color;14;2;[HDR];[Header];Create;True;1;FABRICS;0;0;False;0;False;0.5849056,0.5418971,0.4331613,0;0,0.1132075,0.01206957,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;67;-8518.76,668.0576;Inherit;False;Property;_Exteriorwalls1colour;Exterior walls 1 colour;1;2;[HDR];[Header];Create;True;1;WALLS ;0;0;False;0;False;0.6792453,0.6010633,0.5863296,1;0,0.1793142,0.7264151,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;97;-6631.768,668.1229;Inherit;False;Property;_Wood1color;Wood 1 color;11;2;[HDR];[Header];Create;True;1;WOOD;0;0;False;0;False;0.4056604,0.2683544,0.135858,1;0,0.1793142,0.7264151,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TexturePropertyNode;296;-1237.267,-1838.107;Inherit;True;Property;_Exteriorwallstexture;Exterior walls texture;6;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.TriplanarNode;298;-694.5273,-1744.504;Inherit;True;Cylindrical;World;False;Top Texture 1;_TopTexture1;white;2;None;Mid Texture 1;_MidTexture1;white;6;None;Bot Texture 1;_BotTexture1;white;4;None;Triplanar Sampler;Tangent;10;0;SAMPLER2D;;False;5;FLOAT;1;False;1;SAMPLER2D;;False;6;FLOAT;0;False;2;SAMPLER2D;;False;7;FLOAT;0;False;9;FLOAT3;0,0,0;False;8;FLOAT3;1,1,1;False;3;FLOAT2;1,1;False;4;FLOAT;1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;333;-1047.816,-1546.005;Inherit;False;Constant;_Float3;Float 3;44;0;Create;True;0;0;0;False;0;False;10;0;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;331;-1208.624,-1406.94;Inherit;False;Property;_Exteriorwallstiling;Exterior walls tiling;7;0;Create;True;0;0;0;False;0;False;0;6.2;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;154;-1158.091,706.2426;Inherit;False;Property;_Ropecolor;Rope color;34;2;[HDR];[Header];Create;True;1;OTHER COLORS;0;0;False;0;False;0.6037736,0.5810711,0.3389106,1;0.1698113,0.04637412,0.02963688,1;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;74;1444.886,720.598;Inherit;False;Property;_Metal1color;Metal 1 color;25;2;[HDR];[Header];Create;True;1;METAL;0;0;False;0;False;0.385947,0.5532268,0.5566038,0;0.9528302,0.9528302,0.9528302,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.StaticSwitch;328;6279.643,750.4373;Inherit;False;Property;_InteriortextureOnOff;Interior texture On/Off;8;0;Create;True;0;0;0;False;1;Header(INTERIOR WALLS  DETAILS);False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StaticSwitch;266;7530.528,712.897;Inherit;False;Property;_ExteriortextureOnOff;Exterior texture On/Off;5;0;Create;True;0;0;0;False;1;Header(EXTERIOR WALLS  DETAILS);False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
WireConnection;7;1;254;0
WireConnection;7;3;107;0
WireConnection;9;0;126;0
WireConnection;9;1;6;0
WireConnection;9;2;135;0
WireConnection;10;0;9;0
WireConnection;10;1;8;0
WireConnection;10;2;7;0
WireConnection;11;1;254;0
WireConnection;11;3;106;0
WireConnection;13;1;254;0
WireConnection;13;3;105;0
WireConnection;15;0;10;0
WireConnection;15;1;12;0
WireConnection;15;2;11;0
WireConnection;19;0;15;0
WireConnection;19;1;16;0
WireConnection;19;2;13;0
WireConnection;20;1;302;0
WireConnection;20;3;103;0
WireConnection;22;0;19;0
WireConnection;22;1;18;0
WireConnection;22;2;20;0
WireConnection;25;1;302;0
WireConnection;25;3;104;0
WireConnection;26;1;75;0
WireConnection;26;2;92;0
WireConnection;27;0;22;0
WireConnection;27;1;23;0
WireConnection;27;2;25;0
WireConnection;31;0;26;0
WireConnection;31;1;76;0
WireConnection;31;2;95;0
WireConnection;32;0;27;0
WireConnection;32;1;29;0
WireConnection;32;2;101;0
WireConnection;33;0;247;0
WireConnection;33;1;89;0
WireConnection;34;0;31;0
WireConnection;34;1;77;0
WireConnection;34;2;96;0
WireConnection;39;0;93;0
WireConnection;39;1;35;0
WireConnection;39;2;96;0
WireConnection;41;1;305;0
WireConnection;41;3;38;0
WireConnection;42;0;34;0
WireConnection;42;1;79;0
WireConnection;42;2;41;0
WireConnection;43;0;42;0
WireConnection;43;1;80;0
WireConnection;43;2;116;0
WireConnection;46;0;43;0
WireConnection;46;1;81;0
WireConnection;47;0;72;0
WireConnection;47;1;147;0
WireConnection;47;2;41;0
WireConnection;69;0;248;0
WireConnection;69;1;154;0
WireConnection;70;0;248;0
WireConnection;70;1;73;0
WireConnection;71;0;39;0
WireConnection;71;1;69;0
WireConnection;71;2;87;0
WireConnection;72;0;71;0
WireConnection;72;1;70;0
WireConnection;72;2;86;0
WireConnection;86;1;304;0
WireConnection;86;3;85;0
WireConnection;87;1;304;0
WireConnection;87;3;84;0
WireConnection;91;0;32;0
WireConnection;91;1;30;0
WireConnection;91;2;92;0
WireConnection;92;1;303;0
WireConnection;92;3;17;0
WireConnection;93;0;91;0
WireConnection;93;1;33;0
WireConnection;93;2;95;0
WireConnection;95;1;303;0
WireConnection;95;3;24;0
WireConnection;96;1;303;0
WireConnection;96;3;94;0
WireConnection;101;1;302;0
WireConnection;101;3;102;0
WireConnection;110;1;78;0
WireConnection;110;2;41;0
WireConnection;112;0;110;0
WireConnection;112;1;114;0
WireConnection;112;2;116;0
WireConnection;113;0;112;0
WireConnection;113;1;115;0
WireConnection;113;2;149;0
WireConnection;116;1;305;0
WireConnection;116;3;117;0
WireConnection;122;1;253;0
WireConnection;122;3;121;0
WireConnection;124;0;133;0
WireConnection;124;1;123;0
WireConnection;124;2;122;0
WireConnection;126;0;124;0
WireConnection;126;1;125;0
WireConnection;126;2;137;0
WireConnection;135;1;253;0
WireConnection;135;3;136;0
WireConnection;137;1;253;0
WireConnection;137;3;138;0
WireConnection;148;0;249;0
WireConnection;148;1;111;0
WireConnection;149;1;305;0
WireConnection;149;3;118;0
WireConnection;150;0;47;0
WireConnection;150;1;148;0
WireConnection;150;2;116;0
WireConnection;151;0;150;0
WireConnection;151;1;153;0
WireConnection;151;2;149;0
WireConnection;49;0;157;4
WireConnection;56;1;49;0
WireConnection;56;2;54;0
WireConnection;56;3;51;0
WireConnection;56;4;53;1
WireConnection;56;5;53;2
WireConnection;59;0;56;0
WireConnection;60;0;156;0
WireConnection;62;0;60;0
WireConnection;63;0;59;0
WireConnection;156;0;49;0
WireConnection;55;0;113;0
WireConnection;55;1;50;0
WireConnection;55;2;156;0
WireConnection;123;0;244;0
WireConnection;123;1;127;0
WireConnection;125;0;244;0
WireConnection;125;1;128;0
WireConnection;6;0;244;0
WireConnection;6;1;66;0
WireConnection;8;0;245;0
WireConnection;8;1;67;0
WireConnection;12;0;245;0
WireConnection;12;1;100;0
WireConnection;16;0;245;0
WireConnection;16;1;68;0
WireConnection;18;0;246;0
WireConnection;18;1;97;0
WireConnection;23;0;246;0
WireConnection;23;1;98;0
WireConnection;29;0;246;0
WireConnection;29;1;99;0
WireConnection;30;0;247;0
WireConnection;30;1;88;0
WireConnection;35;0;247;0
WireConnection;35;1;90;0
WireConnection;147;0;249;0
WireConnection;147;1;74;0
WireConnection;153;0;249;0
WireConnection;153;1;152;0
WireConnection;226;0;7;0
WireConnection;226;1;11;0
WireConnection;226;2;13;0
WireConnection;227;0;226;0
WireConnection;243;0;119;0
WireConnection;251;0;120;0
WireConnection;139;0;250;0
WireConnection;139;1;142;0
WireConnection;132;0;313;0
WireConnection;132;1;139;0
WireConnection;132;2;131;0
WireConnection;131;1;252;0
WireConnection;131;3;130;0
WireConnection;140;0;250;0
WireConnection;140;1;143;0
WireConnection;133;0;132;0
WireConnection;133;1;140;0
WireConnection;133;2;141;0
WireConnection;141;1;252;0
WireConnection;141;3;134;0
WireConnection;308;0;318;0
WireConnection;308;1;307;0
WireConnection;309;1;308;0
WireConnection;309;2;310;0
WireConnection;310;1;317;0
WireConnection;310;3;315;0
WireConnection;312;0;318;0
WireConnection;312;1;311;0
WireConnection;313;0;309;0
WireConnection;313;1;312;0
WireConnection;313;2;316;0
WireConnection;316;1;317;0
WireConnection;316;3;314;0
WireConnection;299;0;298;0
WireConnection;329;0;310;0
WireConnection;322;0;330;0
WireConnection;322;1;330;0
WireConnection;322;2;330;0
WireConnection;322;9;320;0
WireConnection;322;3;332;0
WireConnection;323;0;322;0
WireConnection;0;0;266;0
WireConnection;0;3;166;0
WireConnection;0;4;173;0
WireConnection;0;9;171;0
WireConnection;64;0;62;0
WireConnection;65;0;63;0
WireConnection;165;0;151;0
WireConnection;165;1;65;0
WireConnection;165;2;64;0
WireConnection;325;0;327;0
WireConnection;325;1;165;0
WireConnection;325;2;326;0
WireConnection;334;0;331;0
WireConnection;231;0;223;0
WireConnection;231;1;328;0
WireConnection;231;2;228;0
WireConnection;166;1;55;0
WireConnection;173;1;46;0
WireConnection;298;0;296;0
WireConnection;298;1;296;0
WireConnection;298;2;296;0
WireConnection;298;9;295;0
WireConnection;298;3;334;0
WireConnection;298;4;333;0
WireConnection;328;1;165;0
WireConnection;328;0;325;0
WireConnection;266;1;328;0
WireConnection;266;0;231;0
ASEEND*/
//CHKSM=D0210D45CAB88DF95622F078B372BE0E8AA0BBBF