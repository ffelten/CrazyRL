// Made with Amplify Shader Editor v1.9.1.5
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/PT_Vegetation_Opaque_Shader"
{
	Properties
	{
		[NoScaleOffset]_BaseTexture("Base Texture", 2D) = "white" {}
		[Toggle]_CUSTOMCOLORSTINTING("CUSTOM COLORS  TINTING", Float) = 0
		[HDR]_GroundColor("Ground Color", Color) = (0.08490568,0.05234205,0.04846032,1)
		[HDR]_TopColor("Top Color", Color) = (0.4811321,0.4036026,0.2382966,1)
		[HDR]_Gradient("Gradient ", Range( 0 , 1)) = 1
		_GradientPower("Gradient Power", Range( 0 , 10)) = 1
		_Smoothness("Smoothness", Range( 0 , 1)) = 0.7748996
		[Toggle(_SNOWONOFF_ON)] _SNOWONOFF("SNOW ON/OFF", Float) = 0
		_SnowAmount("Snow Amount", Range( 0 , 1)) = 1
		_SnowCoverage("Snow Coverage", Range( 0 , 1)) = 0.45
		_SnowFade("Snow Fade", Range( 0 , 1)) = 0.83
		[Toggle(_CUSTOMWIND_ON)] _CUSTOMWIND("CUSTOM WIND", Float) = 1
		[Toggle(_WINDMASKONOFF_ON)] _WINDMASKONOFF("WIND MASK ON/OFF", Float) = 0
		_WindMovement("Wind Movement", Range( 0 , 10)) = 0.5
		_WindDensity("Wind Density", Range( 0 , 5)) = 3.3
		_WindStrength("Wind Strength", Range( 0 , 1)) = 0.3
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "Opaque"  "Queue" = "Geometry+0" }
		Cull Off
		CGINCLUDE
		#include "UnityShaderVariables.cginc"
		#include "UnityPBSLighting.cginc"
		#include "Lighting.cginc"
		#pragma target 3.5
		#pragma multi_compile_instancing
		#pragma shader_feature _CUSTOMWIND_ON
		#pragma shader_feature_local _WINDMASKONOFF_ON
		#pragma shader_feature_local _SNOWONOFF_ON
		#pragma multi_compile __ LOD_FADE_CROSSFADE
		struct Input
		{
			float2 uv_texcoord;
			float3 worldPos;
			float3 worldNormal;
		};

		uniform float _WindMovement;
		uniform float _WindDensity;
		uniform float _WindStrength;
		uniform float _CUSTOMCOLORSTINTING;
		uniform sampler2D _BaseTexture;
		uniform float4 _GroundColor;
		uniform float4 _TopColor;
		uniform float _Gradient;
		uniform float _GradientPower;
		uniform float _SnowAmount;
		uniform float _SnowFade;
		uniform float _SnowCoverage;
		uniform float _Smoothness;


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


		float4 CalculateContrast( float contrastValue, float4 colorTarget )
		{
			float t = 0.5 * ( 1.0 - contrastValue );
			return mul( float4x4( contrastValue,0,0,t, 0,contrastValue,0,t, 0,0,contrastValue,t, 0,0,0,1 ), colorTarget );
		}

		void vertexDataFunc( inout appdata_full v, out Input o )
		{
			UNITY_INITIALIZE_OUTPUT( Input, o );
			float4 ase_vertex4Pos = v.vertex;
			float simplePerlin2D622 = snoise( (ase_vertex4Pos*1.0 + ( _Time.y * _WindMovement )).xy*_WindDensity );
			simplePerlin2D622 = simplePerlin2D622*0.5 + 0.5;
			float4 appendResult632 = (float4(( ( ( ( simplePerlin2D622 - 0.5 ) / 10.0 ) * _WindStrength ) + ase_vertex4Pos.x ) , ase_vertex4Pos.y , ase_vertex4Pos.z , 1.0));
			#ifdef _WINDMASKONOFF_ON
				float staticSwitch633 = ( ( 1.0 - v.texcoord.xy.y ) * 5.0 );
			#else
				float staticSwitch633 = ase_vertex4Pos.y;
			#endif
			float4 lerpResult635 = lerp( ase_vertex4Pos , appendResult632 , staticSwitch633);
			float4 transform636 = mul(unity_WorldToObject,float4( _WorldSpaceCameraPos , 0.0 ));
			float4 temp_cast_2 = (transform636.w).xxxx;
			#ifdef _CUSTOMWIND_ON
				float4 staticSwitch638 = ( lerpResult635 - temp_cast_2 );
			#else
				float4 staticSwitch638 = ase_vertex4Pos;
			#endif
			float4 LOCALWIND639 = staticSwitch638;
			v.vertex.xyz = LOCALWIND639.xyz;
			v.vertex.w = 1;
		}

		void surf( Input i , inout SurfaceOutputStandard o )
		{
			float2 uv_BaseTexture2 = i.uv_texcoord;
			float4 tex2DNode2 = tex2D( _BaseTexture, uv_BaseTexture2 );
			float4 ase_vertex4Pos = mul( unity_WorldToObject, float4( i.worldPos , 1 ) );
			float clampResult555 = clamp( pow( ( (0.5 + (ase_vertex4Pos.y - 0.0) * (2.0 - 0.5) / (1.0 - 0.0)) * _Gradient ) , _GradientPower ) , 0.0 , 1.0 );
			float4 lerpResult557 = lerp( _GroundColor , _TopColor , clampResult555);
			float4 Gradient558 = lerpResult557;
			float grayscale180 = dot(tex2DNode2.rgb, float3(0.299,0.587,0.114));
			float saferPower568 = abs( grayscale180 );
			float4 temp_cast_1 = (pow( saferPower568 , 0.5 )).xxxx;
			float4 blendOpSrc18 = Gradient558;
			float4 blendOpDest18 = CalculateContrast(1.8,temp_cast_1);
			float2 temp_cast_2 = (0.5).xx;
			float2 uv_TexCoord578 = i.uv_texcoord + temp_cast_2;
			float clampResult596 = clamp( step( uv_TexCoord578.x , 1.0 ) , 0.0 , 1.0 );
			float2 temp_cast_3 = (0.0).xx;
			float2 uv_TexCoord577 = i.uv_texcoord + temp_cast_3;
			float clampResult585 = clamp( ( ( 1.0 - clampResult596 ) * step( uv_TexCoord577.y , 0.5 ) ) , 0.0 , 1.0 );
			float FRUITSMASK586 = clampResult585;
			float temp_output_594_0 = ( 1.0 - FRUITSMASK586 );
			float4 lerpBlendMode18 = lerp(blendOpDest18,( blendOpSrc18 * blendOpDest18 ),temp_output_594_0);
			float4 lerpResult595 = lerp( tex2DNode2 , ( saturate( lerpBlendMode18 )) , temp_output_594_0);
			float4 COLOR502 = (( _CUSTOMCOLORSTINTING )?( lerpResult595 ):( tex2DNode2 ));
			float3 ase_worldPos = i.worldPos;
			float3 ase_worldViewDir = normalize( UnityWorldSpaceViewDir( ase_worldPos ) );
			float3 ase_worldNormal = i.worldNormal;
			float4 color443 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			float fresnelNdotV454 = dot( ase_worldNormal, ase_worldViewDir );
			float fresnelNode454 = ( 0.11 + 1.0 * pow( 1.0 - fresnelNdotV454, color443.r ) );
			float3 ase_normWorldNormal = normalize( ase_worldNormal );
			float dotResult450 = dot( ase_normWorldNormal , float3(0,1,0) );
			float smoothstepResult531 = smoothstep( 0.0 , _SnowFade , ( dotResult450 + (-1.0 + (_SnowCoverage - 0.0) * (1.0 - -1.0) / (1.0 - 0.0)) ));
			float SNOW489 = ( ( (0.0 + (_SnowAmount - 0.0) * (10.0 - 0.0) / (1.0 - 0.0)) * fresnelNode454 ) * smoothstepResult531 );
			#ifdef _SNOWONOFF_ON
				float4 staticSwitch372 = ( SNOW489 + COLOR502 );
			#else
				float4 staticSwitch372 = COLOR502;
			#endif
			o.Albedo = staticSwitch372.rgb;
			o.Smoothness = ( _Smoothness * ( 1.0 - FRUITSMASK586 ) );
			o.Alpha = 1;
		}

		ENDCG
		CGPROGRAM
		#pragma surface surf Standard keepalpha fullforwardshadows exclude_path:deferred dithercrossfade vertex:vertexDataFunc 

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
				float2 customPack1 : TEXCOORD1;
				float3 worldPos : TEXCOORD2;
				float3 worldNormal : TEXCOORD3;
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
				vertexDataFunc( v, customInputData );
				float3 worldPos = mul( unity_ObjectToWorld, v.vertex ).xyz;
				half3 worldNormal = UnityObjectToWorldNormal( v.normal );
				o.worldNormal = worldNormal;
				o.customPack1.xy = customInputData.uv_texcoord;
				o.customPack1.xy = v.texcoord;
				o.worldPos = worldPos;
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
				float3 worldPos = IN.worldPos;
				half3 worldViewDir = normalize( UnityWorldSpaceViewDir( worldPos ) );
				surfIN.worldPos = worldPos;
				surfIN.worldNormal = IN.worldNormal;
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
Node;AmplifyShaderEditor.CommentaryNode;574;2759.948,124.1273;Inherit;False;1868.886;616.2441;mask;13;586;585;584;582;581;580;579;578;577;576;592;596;575;;1,1,1,1;0;0
Node;AmplifyShaderEditor.RangedFloatNode;575;2771.608,201.3972;Inherit;False;Constant;_Float0;Float 0;11;0;Create;True;0;0;0;False;0;False;0.5;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;544;-1779.585,-1129.231;Inherit;False;1754.419;983.1141;GRADIENT;10;557;556;553;555;572;547;571;573;546;545;GRADIENT;1,1,1,1;0;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;578;3073.636,169.6081;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;579;2972.972,336.3389;Float;False;Constant;_Float3;Float 3;11;0;Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;576;2773.524,508.5035;Inherit;False;Constant;_Float1;Float 1;11;0;Create;True;0;0;0;False;0;False;0;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.StepOpNode;581;3470.578,185.767;Inherit;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;545;-1765.844,-694.5081;Inherit;False;1;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TextureCoordinatesNode;577;3076.713,458.9164;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ClampOpNode;596;3709.03,177.3771;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;546;-1750.859,-441.2354;Float;False;Property;_Gradient;Gradient ;7;1;[HDR];Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;580;3079.748,637.1544;Inherit;False;Constant;_Float5;Float 5;11;0;Create;True;0;0;0;False;0;False;0.5;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;573;-1580.877,-655.3626;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0.5;False;4;FLOAT;2;False;1;FLOAT;0
Node;AmplifyShaderEditor.StepOpNode;582;3340.018,429.8082;Inherit;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;39;-2704.9,-37.7472;Inherit;False;2729.862;955.7487;COLOR;12;502;336;18;352;180;2;127;567;568;593;594;595;COLOR;1,1,1,1;0;0
Node;AmplifyShaderEditor.OneMinusNode;584;4098.711,224.6663;Inherit;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;547;-1374.914,-680.115;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;571;-1391.498,-481.5369;Inherit;False;Property;_GradientPower;Gradient Power;8;0;Create;True;0;0;0;False;0;False;1;10;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;127;-2616.817,68.34903;Inherit;True;Property;_BaseTexture;Base Texture;0;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.PowerNode;572;-1167.086,-689.5126;Inherit;True;False;2;0;FLOAT;0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;592;3747.828,403.7969;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ClampOpNode;555;-882.071,-707.351;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SamplerNode;2;-2366.745,67.82262;Inherit;True;Property;_TextureSample0;Texture Sample 0;1;0;Create;True;0;0;0;False;0;False;-1;None;None;True;0;False;black;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ClampOpNode;585;4018.493,431.1993;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;556;-1680.195,-881.4604;Float;False;Property;_GroundColor;Ground Color;3;1;[HDR];Create;True;0;0;0;False;0;False;0.08490568,0.05234205,0.04846032,1;0.01743852,0.5754717,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;553;-1689.1,-1070.706;Float;False;Property;_TopColor;Top Color;5;1;[HDR];Create;True;0;0;0;False;0;False;0.4811321,0.4036026,0.2382966,1;0.05298166,0.3490566,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;557;-585.7902,-984.1019;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;363;-792.7774,2558.05;Inherit;False;1693.406;1367.284;Comment;15;489;441;467;458;531;532;452;454;535;446;455;443;445;450;442;SNOW;1,1,1,1;0;0
Node;AmplifyShaderEditor.TFHCGrayscale;180;-2001.596,176.2859;Inherit;True;1;1;0;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;586;4335.603,399.908;Inherit;True;FRUITSMASK;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;558;-75.64561,-952.1125;Inherit;False;Gradient;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;593;-1642.487,454.9655;Inherit;True;586;FRUITSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.PowerNode;568;-1764.454,188.4758;Inherit;True;True;2;0;FLOAT;0;False;1;FLOAT;0.5;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldNormalVector;441;-765.6613,3118.918;Inherit;True;True;1;0;FLOAT3;0,0,1;False;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.RangedFloatNode;445;-340.7266,3385.052;Inherit;True;Property;_SnowCoverage;Snow Coverage;14;0;Create;True;0;0;0;False;0;False;0.45;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.Vector3Node;442;-563.7465,3344.3;Inherit;False;Constant;_SnowDirection;Snow Direction;11;0;Create;True;0;0;0;False;0;False;0,1,0;0,1,0;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.DotProductOpNode;450;-278.7435,3091.426;Inherit;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;352;-1453.641,37.96606;Inherit;False;558;Gradient;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleContrastOpNode;567;-1507.181,190.6048;Inherit;True;2;1;COLOR;0,0,0,0;False;0;FLOAT;1.8;False;1;COLOR;0
Node;AmplifyShaderEditor.OneMinusNode;594;-1399.428,449.284;Inherit;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;443;-751.9623,2912.807;Inherit;False;Constant;_Color1;Color 1;30;0;Create;True;0;0;0;False;0;False;1,1,1,0;1,1,1,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.TFHCRemapNode;455;-7.676476,3387.507;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;-1;False;4;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;446;-679.6005,2650.073;Inherit;False;Property;_SnowAmount;Snow Amount;13;0;Create;True;0;0;0;False;0;False;1;0.82;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.BlendOpsNode;18;-1216.806,141.9453;Inherit;True;Multiply;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;1;False;1;COLOR;0
Node;AmplifyShaderEditor.TFHCRemapNode;452;-291.1125,2661.683;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0;False;4;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;532;-28.19747,2942.418;Inherit;False;Property;_SnowFade;Snow Fade;15;0;Create;True;0;0;0;False;0;False;0.83;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;535;238.4201,3349.512;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.FresnelNode;454;-518.5674,2824.515;Inherit;False;Standard;WorldNormal;ViewDir;True;False;5;0;FLOAT3;0,0,1;False;4;FLOAT3;0,0,0;False;1;FLOAT;0.11;False;2;FLOAT;1;False;3;FLOAT;5;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;595;-883.1122,290.2584;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;458;-36.43948,2775.072;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SmoothstepOpNode;531;333.4893,2983.057;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;336;-630.8104,37.76899;Inherit;True;Property;_CUSTOMCOLORSTINTING;CUSTOM COLORS  TINTING;1;0;Create;True;0;0;0;False;0;False;0;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;467;464.95,2789.231;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;489;640.451,2773.928;Inherit;True;SNOW;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;502;-200.9671,38.16975;Inherit;True;COLOR;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;366;886.0975,403.3277;Inherit;False;489;SNOW;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;367;823.0651,115.0585;Inherit;True;502;COLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;599;1332.476,562.7034;Inherit;False;586;FRUITSMASK;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;600;1540.107,584.6141;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;368;1111.06,296.3205;Inherit;False;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;566;1395.757,413.0894;Inherit;False;Property;_Smoothness;Smoothness;10;0;Create;True;0;0;0;False;0;False;0.7748996;0.48;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;598;1710.176,434.369;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;607;-1283.442,-2134.64;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;606;-1405.483,-1845.012;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;612;-1111.067,-2743.006;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;130;2085.109,83.74197;Inherit;False;Property;_MaskClipValue;Mask Clip Value;11;1;[HideInInspector];Fetch;True;0;0;0;False;0;False;0.5;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;610;-2012.019,-2537.085;Float;False;Property;_MIDDLECOLOR;MIDDLE COLOR;4;0;Create;True;0;0;0;False;0;False;0,1,0.819211,1;0,0.5943396,0.0169811,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ClampOpNode;609;-1220.379,-2448.365;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;613;-2009.902,-2261.715;Float;False;Property;_TOPCOLOR;TOP COLOR;2;0;Create;True;0;0;0;False;0;False;1,0.02087107,0,1;0.01743852,0.5754717,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;608;-2002.034,-2762.489;Float;False;Property;_GROUNDCOLOR;GROUND COLOR;6;0;Create;True;0;0;0;False;0;False;0.9778857,1,0,1;0.05298166,0.3490566,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;614;-611.145,-2305.529;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ClampOpNode;611;-1100.958,-1841.228;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;605;-1610.876,-2098.784;Inherit;True;5;0;FLOAT;0;False;1;FLOAT;-0.2;False;2;FLOAT;0.4;False;3;FLOAT;0;False;4;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;601;-2078.584,-1854.304;Float;False;Property;_COLORGRADIENTRATIO;COLOR GRADIENT RATIO;9;1;[HDR];Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;604;-1617.611,-1840.008;Inherit;True;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0.5;False;3;FLOAT;0;False;4;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;602;-2033.996,-2032.432;Inherit;False;0;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;603;-1785.734,-1978.1;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.StaticSwitch;372;1282.186,125.3575;Inherit;True;Property;_SNOWONOFF;SNOW ON/OFF;12;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;62;2050.875,301.853;Float;False;True;-1;3;;0;0;Standard;Polytope Studio/PT_Vegetation_Opaque_Shader;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;True;False;False;False;False;Off;0;False;;0;False;;False;0;False;;0;False;;False;0;Opaque;0.5;True;True;0;False;Opaque;;Geometry;ForwardOnly;12;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;False;2;15;10;25;False;0.5;True;0;5;False;;10;False;;0;0;False;;0;False;;0;False;;0;False;;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Absolute;0;;-1;-1;-1;-1;0;False;0;0;False;;-1;0;True;_MaskClipValue;1;Pragma;multi_compile __ LOD_FADE_CROSSFADE;False;;Custom;False;0;0;;0;0;False;0.1;False;;0;False;;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
Node;AmplifyShaderEditor.CommentaryNode;615;-4551.278,1265.143;Inherit;False;3563.478;765.4585;WIND;24;639;638;637;636;635;634;633;632;631;630;629;628;627;626;625;624;623;622;621;620;619;618;617;616;WIND;1,1,1,1;0;0
Node;AmplifyShaderEditor.SimpleTimeNode;616;-4455.682,1594.339;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;617;-4526.1,1723.463;Inherit;False;Property;_WindMovement;Wind Movement;18;0;Create;True;0;0;0;False;0;False;0.5;0.5;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;618;-4261.693,1597.278;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;619;-4354.106,1301.285;Inherit;False;1;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;620;-4207.618,1726.763;Inherit;False;Property;_WindDensity;Wind Density;19;0;Create;True;0;0;0;False;0;False;3.3;1.91;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.ScaleAndOffsetNode;621;-4118.006,1491.76;Inherit;False;3;0;FLOAT4;0,0,0,0;False;1;FLOAT;1;False;2;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;622;-3890.281,1503.992;Inherit;False;Simplex2D;True;False;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;623;-3684.491,1516.59;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0.5;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleDivideOpNode;624;-3544.645,1511.105;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.TexCoordVertexDataNode;625;-4008.122,1832.116;Inherit;False;0;2;0;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;626;-3877.751,1750.948;Inherit;False;Property;_WindStrength;Wind Strength;20;0;Create;True;0;0;0;False;0;False;0.3;0.203;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;627;-3751.038,1871.178;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;628;-3419.844,1509.483;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;629;-3971.92,2026.42;Inherit;False;Constant;_VertexMask;Vertex Mask;21;0;Create;True;0;0;0;False;0;False;5;0.5;-5;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;630;-3506.188,1751.684;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;631;-3273.152,1412.394;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.DynamicAppendNode;632;-3023.009,1353.298;Inherit;True;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;1;False;1;FLOAT4;0
Node;AmplifyShaderEditor.StaticSwitch;633;-3264.064,1698.952;Inherit;False;Property;_WINDMASKONOFF;WIND MASK ON/OFF;17;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;All;9;1;FLOAT;0;False;0;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT;0;False;7;FLOAT;0;False;8;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldSpaceCameraPos;634;-2902.26,1654.427;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.LerpOp;635;-2778.247,1368.13;Inherit;True;3;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0,0,0,0;False;2;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.WorldToObjectTransfNode;636;-2653.194,1647.678;Inherit;False;1;0;FLOAT4;0,0,0,1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleSubtractOpNode;637;-2480.229,1386.635;Inherit;True;2;0;FLOAT4;0,0,0,0;False;1;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.StaticSwitch;638;-2237.332,1298.908;Inherit;True;Property;_CUSTOMWIND;CUSTOM WIND;16;0;Create;True;0;0;0;False;0;False;0;1;1;True;;Toggle;2;Key0;Key1;Create;False;True;All;9;1;FLOAT4;0,0,0,0;False;0;FLOAT4;0,0,0,0;False;2;FLOAT4;0,0,0,0;False;3;FLOAT4;0,0,0,0;False;4;FLOAT4;0,0,0,0;False;5;FLOAT4;0,0,0,0;False;6;FLOAT4;0,0,0,0;False;7;FLOAT4;0,0,0,0;False;8;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;639;-1492.558,1301.6;Inherit;True;LOCALWIND;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;354;1641.764,673.1421;Inherit;False;639;LOCALWIND;1;0;OBJECT;;False;1;FLOAT4;0
WireConnection;578;1;575;0
WireConnection;581;0;578;1
WireConnection;581;1;579;0
WireConnection;577;1;576;0
WireConnection;596;0;581;0
WireConnection;573;0;545;2
WireConnection;582;0;577;2
WireConnection;582;1;580;0
WireConnection;584;0;596;0
WireConnection;547;0;573;0
WireConnection;547;1;546;0
WireConnection;572;0;547;0
WireConnection;572;1;571;0
WireConnection;592;0;584;0
WireConnection;592;1;582;0
WireConnection;555;0;572;0
WireConnection;2;0;127;0
WireConnection;585;0;592;0
WireConnection;557;0;556;0
WireConnection;557;1;553;0
WireConnection;557;2;555;0
WireConnection;180;0;2;0
WireConnection;586;0;585;0
WireConnection;558;0;557;0
WireConnection;568;0;180;0
WireConnection;450;0;441;0
WireConnection;450;1;442;0
WireConnection;567;1;568;0
WireConnection;594;0;593;0
WireConnection;455;0;445;0
WireConnection;18;0;352;0
WireConnection;18;1;567;0
WireConnection;18;2;594;0
WireConnection;452;0;446;0
WireConnection;535;0;450;0
WireConnection;535;1;455;0
WireConnection;454;3;443;0
WireConnection;595;0;2;0
WireConnection;595;1;18;0
WireConnection;595;2;594;0
WireConnection;458;0;452;0
WireConnection;458;1;454;0
WireConnection;531;0;535;0
WireConnection;531;2;532;0
WireConnection;336;0;2;0
WireConnection;336;1;595;0
WireConnection;467;0;458;0
WireConnection;467;1;531;0
WireConnection;489;0;467;0
WireConnection;502;0;336;0
WireConnection;600;0;599;0
WireConnection;368;0;366;0
WireConnection;368;1;367;0
WireConnection;598;0;566;0
WireConnection;598;1;600;0
WireConnection;607;0;606;0
WireConnection;607;1;605;0
WireConnection;606;0;604;0
WireConnection;612;0;608;0
WireConnection;612;1;610;0
WireConnection;612;2;609;0
WireConnection;609;0;607;0
WireConnection;614;0;612;0
WireConnection;614;1;613;0
WireConnection;614;2;611;0
WireConnection;611;0;604;0
WireConnection;605;0;603;0
WireConnection;604;0;603;0
WireConnection;603;0;602;2
WireConnection;603;1;601;0
WireConnection;372;1;367;0
WireConnection;372;0;368;0
WireConnection;62;0;372;0
WireConnection;62;4;598;0
WireConnection;62;11;354;0
WireConnection;618;0;616;0
WireConnection;618;1;617;0
WireConnection;621;0;619;0
WireConnection;621;2;618;0
WireConnection;622;0;621;0
WireConnection;622;1;620;0
WireConnection;623;0;622;0
WireConnection;624;0;623;0
WireConnection;627;0;625;2
WireConnection;628;0;624;0
WireConnection;628;1;626;0
WireConnection;630;0;627;0
WireConnection;630;1;629;0
WireConnection;631;0;628;0
WireConnection;631;1;619;1
WireConnection;632;0;631;0
WireConnection;632;1;619;2
WireConnection;632;2;619;3
WireConnection;633;1;619;2
WireConnection;633;0;630;0
WireConnection;635;0;619;0
WireConnection;635;1;632;0
WireConnection;635;2;633;0
WireConnection;636;0;634;0
WireConnection;637;0;635;0
WireConnection;637;1;636;4
WireConnection;638;1;619;0
WireConnection;638;0;637;0
WireConnection;639;0;638;0
ASEEND*/
//CHKSM=1B05D4CCB70E66F5E7FFE02C1C2280A0C4E37797