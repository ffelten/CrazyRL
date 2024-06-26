// Made with Amplify Shader Editor
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/PT_Vegetation_Plants_Shader"
{
	Properties
	{
		[NoScaleOffset]_BaseTexture("Base Texture", 2D) = "white" {}
		[Toggle]_CUSTOMCOLORSTINTING("CUSTOM COLORS  TINTING", Float) = 0
		[HDR]_TopColor("Top Color", Color) = (0,0.2178235,1,1)
		[HDR]_GroundColor("Ground Color", Color) = (1,0,0,1)
		[HDR]_Gradient("Gradient", Range( 0 , 10)) = 1.4
		_GradientPower("Gradient Power", Range( 0 , 10)) = 1
		_LeavesThickness("Leaves Thickness", Range( 0.1 , 0.95)) = 0.5
		_Smoothness("Smoothness", Range( 0 , 1)) = 0
		[Toggle(_TRANSLUCENCYONOFF_ON)] _TRANSLUCENCYONOFF("TRANSLUCENCY ON/OFF", Float) = 1
		[Header(Translucency)]
		_Translucency("Strength", Range( 0 , 50)) = 1
		_TransNormalDistortion("Normal Distortion", Range( 0 , 1)) = 0.1
		_TransScattering("Scaterring Falloff", Range( 1 , 50)) = 2
		_TransDirect("Direct", Range( 0 , 1)) = 1
		_TransAmbient("Ambient", Range( 0 , 1)) = 0.2
		_TransShadow("Shadow", Range( 0 , 1)) = 0.9
		[Toggle(_CUSTOMWIND_ON)] _CUSTOMWIND("CUSTOM WIND", Float) = 1
		[HideInInspector]_MaskClipValue("Mask Clip Value", Range( 0 , 1)) = 0.5
		_WindMovement("Wind Movement", Range( 0 , 10)) = 0.5
		_WindDensity("Wind Density", Range( 0 , 5)) = 3.3
		_WindStrength("Wind Strength", Range( 0 , 1)) = 0.3
		[Toggle(_SNOWONOFF_ON)] _SNOWONOFF("SNOW ON/OFF", Float) = 0
		_SnowGradient("Snow Gradient", Range( 0 , 1)) = 0.83
		_SnowCoverage("Snow Coverage", Range( 0 , 1)) = 0.45
		_SnowAmount("Snow Amount", Range( 0 , 1)) = 1
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "TransparentCutout"  "Queue" = "Geometry+0" }
		Cull Off
		CGINCLUDE
		#include "UnityShaderVariables.cginc"
		#include "UnityPBSLighting.cginc"
		#include "Lighting.cginc"
		#pragma target 3.5
		#pragma multi_compile_instancing
		#pragma shader_feature _CUSTOMWIND_ON
		#pragma shader_feature_local _SNOWONOFF_ON
		#pragma shader_feature_local _TRANSLUCENCYONOFF_ON
		#pragma multi_compile __ LOD_FADE_CROSSFADE
		struct Input
		{
			float2 uv_texcoord;
			float3 worldPos;
			float3 worldNormal;
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
		uniform sampler2D _BaseTexture;
		uniform float4 _GroundColor;
		uniform float4 _TopColor;
		uniform float _Gradient;
		uniform float _GradientPower;
		uniform float _SnowAmount;
		uniform float _SnowGradient;
		uniform float _SnowCoverage;
		uniform float _Smoothness;
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
			float3 ase_vertex3Pos = v.vertex.xyz;
			float simplePerlin2D308 = snoise( (ase_vertex3Pos*1.0 + ( _Time.y * _WindMovement )).xy*_WindDensity );
			simplePerlin2D308 = simplePerlin2D308*0.5 + 0.5;
			float4 appendResult316 = (float4(( ( ( ( simplePerlin2D308 - 0.5 ) / 10.0 ) * _WindStrength ) + ase_vertex3Pos.x ) , ase_vertex3Pos.y , ase_vertex3Pos.z , 1.0));
			float4 lerpResult317 = lerp( float4( ase_vertex3Pos , 0.0 ) , appendResult316 , ( ase_vertex3Pos.y * 2.0 ));
			float4 transform318 = mul(unity_WorldToObject,float4( _WorldSpaceCameraPos , 0.0 ));
			float4 temp_cast_4 = (transform318.w).xxxx;
			#ifdef _CUSTOMWIND_ON
				float4 staticSwitch320 = ( lerpResult317 - temp_cast_4 );
			#else
				float4 staticSwitch320 = float4( ase_vertex3Pos , 0.0 );
			#endif
			float4 LOCALWIND353 = staticSwitch320;
			v.vertex.xyz = LOCALWIND353.xyz;
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
			float2 uv_BaseTexture2 = i.uv_texcoord;
			float4 tex2DNode2 = tex2D( _BaseTexture, uv_BaseTexture2 );
			float clampResult738 = clamp( pow( ( i.uv_texcoord.y * _Gradient ) , _GradientPower ) , 0.0 , 1.0 );
			float4 lerpResult557 = lerp( _GroundColor , _TopColor , clampResult738);
			float4 GRADIENT558 = lerpResult557;
			float4 blendOpSrc18 = tex2DNode2;
			float4 blendOpDest18 = GRADIENT558;
			float4 lerpBlendMode18 = lerp(blendOpDest18,( blendOpSrc18 * blendOpDest18 ),0.0);
			float4 COLOR502 = (( _CUSTOMCOLORSTINTING )?( lerpBlendMode18 ):( tex2DNode2 ));
			float3 ase_worldPos = i.worldPos;
			float3 ase_worldViewDir = normalize( UnityWorldSpaceViewDir( ase_worldPos ) );
			float3 ase_worldNormal = i.worldNormal;
			float4 color443 = IsGammaSpace() ? float4(1,1,1,0) : float4(1,1,1,0);
			float fresnelNdotV454 = dot( ase_worldNormal, ase_worldViewDir );
			float fresnelNode454 = ( 0.11 + 1.0 * pow( 1.0 - fresnelNdotV454, color443.r ) );
			float smoothstepResult531 = smoothstep( 0.0 , _SnowGradient , ( ( 1.0 - ( i.uv_texcoord.y * 0.65 ) ) + (-1.0 + (_SnowCoverage - 0.0) * (1.0 - -1.0) / (1.0 - 0.0)) ));
			float SNOW489 = ( ( (0.0 + (_SnowAmount - 0.0) * (10.0 - 0.0) / (1.0 - 0.0)) * fresnelNode454 ) * smoothstepResult531 );
			float4 temp_cast_1 = (( SNOW489 + 0.0 )).xxxx;
			#ifdef _SNOWONOFF_ON
				float4 staticSwitch372 = temp_cast_1;
			#else
				float4 staticSwitch372 = COLOR502;
			#endif
			o.Albedo = staticSwitch372.rgb;
			o.Smoothness = _Smoothness;
			#ifdef _TRANSLUCENCYONOFF_ON
				float4 staticSwitch493 = ( COLOR502 * 1.0 );
			#else
				float4 staticSwitch493 = float4( 0,0,0,0 );
			#endif
			float4 TRANSLUCENCY497 = staticSwitch493;
			o.Translucency = TRANSLUCENCY497.rgb;
			o.Alpha = 1;
			float GENERALALPHA505 = tex2DNode2.a;
			float ALPHACUTOFF496 = ( 1.0 - step( GENERALALPHA505 , ( 1.0 - _LeavesThickness ) ) );
			clip( ALPHACUTOFF496 - _MaskClipValue );
		}

		ENDCG
		CGPROGRAM
		#pragma surface surf StandardCustom keepalpha fullforwardshadows exclude_path:deferred dithercrossfade vertex:vertexDataFunc 

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
				SurfaceOutputStandardCustom o;
				UNITY_INITIALIZE_OUTPUT( SurfaceOutputStandardCustom, o )
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
	CustomEditor "ASEMaterialInspector"
}
/*ASEBEGIN
Version=18912
0;0;1920;1029;513.4656;527.3048;2.93292;True;False
Node;AmplifyShaderEditor.CommentaryNode;544;-1538.46,-972.0139;Inherit;False;1541.214;770.4899;GRADIENT;9;558;557;738;556;553;547;546;745;746;GRADIENT;1,1,1,1;0;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;743;-1636.497,-551.8318;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;546;-1654.349,-393.8626;Float;False;Property;_Gradient;Gradient;4;1;[HDR];Create;True;0;0;0;False;0;False;1.4;1;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;301;-3404.303,1022.545;Inherit;False;3563.478;765.4585;WIND;21;319;318;317;314;316;315;313;312;311;309;310;308;307;306;304;305;302;303;320;353;744;WIND;1,1,1,1;0;0
Node;AmplifyShaderEditor.RangedFloatNode;746;-1283.241,-276.4082;Inherit;False;Property;_GradientPower;Gradient Power;5;0;Create;True;0;0;0;False;0;False;1;10;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;547;-1349.036,-524.4625;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;302;-3379.124,1480.865;Inherit;False;Property;_WindMovement;Wind Movement;18;0;Create;True;0;0;0;False;0;False;0.5;0.5;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleTimeNode;303;-3308.706,1351.741;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.PosVertexDataNode;305;-3306.657,1081.432;Inherit;False;0;0;5;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.PowerNode;745;-1058.829,-484.3837;Inherit;True;False;2;0;FLOAT;0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;304;-3114.718,1354.68;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;307;-3060.643,1484.165;Inherit;False;Property;_WindDensity;Wind Density;19;0;Create;True;0;0;0;False;0;False;3.3;1.91;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;556;-1378.864,-750.9102;Float;False;Property;_GroundColor;Ground Color;3;1;[HDR];Create;True;0;0;0;False;0;False;1,0,0,1;0.01743852,0.5754717,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ClampOpNode;738;-862.7357,-535.2225;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;553;-1360.817,-937.2668;Float;False;Property;_TopColor;Top Color;2;1;[HDR];Create;True;0;0;0;False;0;False;0,0.2178235,1,1;0.05298166,0.3490566,0,1;False;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ScaleAndOffsetNode;306;-2971.031,1249.162;Inherit;False;3;0;FLOAT3;0,0,0;False;1;FLOAT;1;False;2;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.CommentaryNode;39;-2704.9,-111.1786;Inherit;False;2726.816;529.2971;COLOR;9;505;502;336;18;352;728;180;2;127;COLOR;1,1,1,1;0;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;308;-2743.306,1261.394;Inherit;False;Simplex2D;True;False;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.LerpOp;557;-602.1736,-865.377;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.CommentaryNode;363;-1660.093,1917.527;Inherit;False;1693.406;1367.284;Comment;20;489;441;467;458;531;532;452;454;535;446;455;443;530;533;445;450;442;528;529;527;SNOW;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;558;-250.0819,-831.2172;Inherit;False;GRADIENT;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;529;-1476.111,3139.16;Inherit;False;Constant;_Float0;Float 0;22;0;Create;True;0;0;0;False;0;False;0.65;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;309;-2544.29,1265.992;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0.5;False;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;127;-2684.817,-34.12083;Inherit;True;Property;_BaseTexture;Base Texture;0;1;[NoScaleOffset];Create;True;0;0;0;False;0;False;None;None;False;white;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.TextureCoordinatesNode;527;-1439.001,2972.903;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;445;-1208.042,2744.531;Inherit;True;Property;_SnowCoverage;Snow Coverage;25;0;Create;True;0;0;0;False;0;False;0.45;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;352;-1372.463,81.00999;Inherit;False;558;GRADIENT;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.SamplerNode;2;-2291.393,-5.608733;Inherit;True;Property;_TextureSample0;Texture Sample 0;1;0;Create;True;0;0;0;False;0;False;-1;None;None;True;0;False;black;Auto;False;Object;-1;Auto;Texture2D;8;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;6;FLOAT;0;False;7;SAMPLERSTATE;;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;310;-2689.176,1370.55;Inherit;False;Property;_WindStrength;Wind Strength;20;0;Create;True;0;0;0;False;0;False;0.3;0.203;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;528;-1139.862,3021.589;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleDivideOpNode;744;-2408.589,1269.884;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;311;-2280.998,1268.111;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.TFHCRemapNode;455;-874.9898,2746.986;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;-1;False;4;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;446;-1546.916,2009.55;Inherit;False;Property;_SnowAmount;Snow Amount;26;0;Create;True;0;0;0;False;0;False;1;0.82;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.BlendOpsNode;18;-1189.932,-13.21011;Inherit;True;Multiply;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.OneMinusNode;533;-935.4996,3034.419;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;443;-1619.278,2272.284;Inherit;False;Constant;_Color1;Color 1;30;0;Create;True;0;0;0;False;0;False;1,1,1,0;1,1,1,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;312;-2411.61,1422.472;Inherit;False;Constant;_Float4;Float 4;7;0;Create;True;0;0;0;False;0;False;2;0;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;535;-628.894,2708.991;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;336;-863.1887,-66.5021;Inherit;False;Property;_CUSTOMCOLORSTINTING;CUSTOM COLORS  TINTING;1;0;Create;True;0;0;0;False;0;False;0;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.FresnelNode;454;-1385.883,2183.992;Inherit;False;Standard;WorldNormal;ViewDir;True;False;5;0;FLOAT3;0,0,1;False;4;FLOAT3;0,0,0;False;1;FLOAT;0.11;False;2;FLOAT;1;False;3;FLOAT;5;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;532;-895.5118,2301.897;Inherit;False;Property;_SnowGradient;Snow Gradient;24;0;Create;True;0;0;0;False;0;False;0.83;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;741;-1203.495,556.3467;Inherit;False;1198.91;343.9196;Comment;6;106;128;107;116;740;496;ALPHA;1,1,1,1;0;0
Node;AmplifyShaderEditor.TFHCRemapNode;452;-1158.428,2021.16;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;0;False;4;FLOAT;10;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;313;-2133.752,1265.846;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;458;-903.7538,2134.55;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.DynamicAppendNode;316;-1871.734,1133.275;Inherit;True;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;1;False;1;FLOAT4;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;315;-2101.593,1398.846;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;365;-1053.508,3402.887;Inherit;False;1092.364;358.1904;Comment;5;497;493;486;491;481;TRANSLUCENCY;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;505;-2001.525,112.4598;Inherit;False;GENERALALPHA;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;502;-200.9671,-35.26162;Inherit;False;COLOR;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.WorldSpaceCameraPos;314;-1755.286,1411.829;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.RangedFloatNode;106;-1153.495,673.434;Inherit;False;Property;_LeavesThickness;Leaves Thickness;6;0;Create;True;0;0;0;False;0;False;0.5;0.346;0.1;0.95;0;1;FLOAT;0
Node;AmplifyShaderEditor.SmoothstepOpNode;531;-533.8249,2342.536;Inherit;True;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;486;-960.4905,3525.454;Inherit;False;Constant;_Float6;Float 6;15;0;Create;True;0;0;0;False;0;False;1;1;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;128;-860.5824,695.5936;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.WorldToObjectTransfNode;318;-1495.82,1405.08;Inherit;False;1;0;FLOAT4;0,0,0,1;False;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;317;-1631.273,1125.532;Inherit;True;3;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0,0,0,0;False;2;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;467;-402.3637,2148.709;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;481;-988.546,3440.888;Inherit;False;502;COLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;740;-847.2863,606.3467;Inherit;False;505;GENERALALPHA;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;319;-1314.335,1173.657;Inherit;True;2;0;FLOAT4;0,0,0,0;False;1;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.StepOpNode;107;-634.9,638.9446;Inherit;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;491;-703.1473,3498.23;Inherit;False;2;2;0;COLOR;0,0,0,0;False;1;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;489;-226.8627,2133.406;Inherit;True;SNOW;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.StaticSwitch;493;-538.4254,3469.279;Inherit;True;Property;_TRANSLUCENCYONOFF;TRANSLUCENCY ON/OFF;8;0;Create;True;0;0;0;False;0;False;0;1;1;True;;Toggle;2;Key0;Key1;Create;True;True;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StaticSwitch;320;-1063.223,1052.745;Inherit;True;Property;_CUSTOMWIND;CUSTOM WIND;16;0;Create;True;0;0;0;False;0;False;0;1;1;True;;Toggle;2;Key0;Key1;Create;False;True;9;1;FLOAT4;0,0,0,0;False;0;FLOAT4;0,0,0,0;False;2;FLOAT4;0,0,0,0;False;3;FLOAT4;0,0,0,0;False;4;FLOAT4;0,0,0,0;False;5;FLOAT4;0,0,0,0;False;6;FLOAT4;0,0,0,0;False;7;FLOAT4;0,0,0,0;False;8;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;366;1350.336,516.4232;Inherit;False;489;SNOW;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.OneMinusNode;116;-414.9932,646.2662;Inherit;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;368;1567.299,408.4161;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;353;-346.6579,1083.727;Inherit;True;LOCALWIND;-1;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.GetLocalVarNode;367;1311.255,298.9597;Inherit;True;502;COLOR;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;496;-229.5852,647.0934;Inherit;False;ALPHACUTOFF;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;497;-230.6066,3468.484;Inherit;False;TRANSLUCENCY;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.TFHCGrayscale;180;-1783.382,81.43656;Inherit;True;2;1;0;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.StaticSwitch;372;1718.154,296.2032;Inherit;False;Property;_SNOWONOFF;SNOW ON/OFF;21;0;Create;True;0;0;0;False;0;False;0;0;0;True;;Toggle;2;Key0;Key1;Create;True;True;9;1;COLOR;0,0,0,0;False;0;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;COLOR;0,0,0,0;False;4;COLOR;0,0,0,0;False;5;COLOR;0,0,0,0;False;6;COLOR;0,0,0,0;False;7;COLOR;0,0,0,0;False;8;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;354;1546.097,609.1887;Inherit;False;353;LOCALWIND;1;0;OBJECT;;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;130;2085.109,83.74197;Inherit;False;Property;_MaskClipValue;Mask Clip Value;17;1;[HideInInspector];Fetch;True;0;0;0;False;0;False;0.5;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;742;1770.503,401.9955;Inherit;False;Property;_Smoothness;Smoothness;7;0;Create;True;0;0;0;False;0;False;0;0;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleContrastOpNode;728;-1562.835,98.97125;Inherit;True;2;1;COLOR;0,0,0,0;False;0;FLOAT;0.1;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;506;1747.329,546.7998;Inherit;False;496;ALPHACUTOFF;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;507;1750.441,472.6032;Inherit;False;497;TRANSLUCENCY;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.DotProductOpNode;450;-1146.058,2450.905;Inherit;True;2;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ToggleSwitchNode;530;-888.5378,2546.178;Inherit;False;Property;_UVSNOW;UV SNOW;22;0;Create;True;0;0;0;False;0;False;0;True;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.Vector3Node;442;-1431.062,2703.779;Inherit;False;Property;_SnowDirection;Snow Direction;23;0;Create;True;0;0;0;False;0;False;0,1,0;0,1,0;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.WorldNormalVector;441;-1632.977,2478.397;Inherit;True;True;1;0;FLOAT3;0,0,1;False;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;62;2050.875,301.853;Float;False;True;-1;3;ASEMaterialInspector;0;0;Standard;Polytope Studio/PT_Vegetation_Plants_Shader;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;True;False;False;False;False;Off;0;False;-1;0;False;-1;False;0;False;-1;0;False;-1;False;0;Custom;0.5;True;True;0;True;TransparentCutout;;Geometry;ForwardOnly;18;all;True;True;True;True;0;False;-1;False;0;False;-1;255;False;-1;255;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;False;2;15;10;25;False;0.5;True;0;5;False;-1;10;False;-1;0;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;Absolute;0;;-1;9;-1;-1;0;False;0;0;False;-1;-1;0;True;130;1;Pragma;multi_compile __ LOD_FADE_CROSSFADE;False;;Custom;0;0;False;0.1;False;-1;0;False;-1;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
WireConnection;547;0;743;2
WireConnection;547;1;546;0
WireConnection;745;0;547;0
WireConnection;745;1;746;0
WireConnection;304;0;303;0
WireConnection;304;1;302;0
WireConnection;738;0;745;0
WireConnection;306;0;305;0
WireConnection;306;2;304;0
WireConnection;308;0;306;0
WireConnection;308;1;307;0
WireConnection;557;0;556;0
WireConnection;557;1;553;0
WireConnection;557;2;738;0
WireConnection;558;0;557;0
WireConnection;309;0;308;0
WireConnection;2;0;127;0
WireConnection;528;0;527;2
WireConnection;528;1;529;0
WireConnection;744;0;309;0
WireConnection;311;0;744;0
WireConnection;311;1;310;0
WireConnection;455;0;445;0
WireConnection;18;0;2;0
WireConnection;18;1;352;0
WireConnection;533;0;528;0
WireConnection;535;0;533;0
WireConnection;535;1;455;0
WireConnection;336;0;2;0
WireConnection;336;1;18;0
WireConnection;454;3;443;0
WireConnection;452;0;446;0
WireConnection;313;0;311;0
WireConnection;313;1;305;1
WireConnection;458;0;452;0
WireConnection;458;1;454;0
WireConnection;316;0;313;0
WireConnection;316;1;305;2
WireConnection;316;2;305;3
WireConnection;315;0;305;2
WireConnection;315;1;312;0
WireConnection;505;0;2;4
WireConnection;502;0;336;0
WireConnection;531;0;535;0
WireConnection;531;2;532;0
WireConnection;128;0;106;0
WireConnection;318;0;314;0
WireConnection;317;0;305;0
WireConnection;317;1;316;0
WireConnection;317;2;315;0
WireConnection;467;0;458;0
WireConnection;467;1;531;0
WireConnection;319;0;317;0
WireConnection;319;1;318;4
WireConnection;107;0;740;0
WireConnection;107;1;128;0
WireConnection;491;0;481;0
WireConnection;491;1;486;0
WireConnection;489;0;467;0
WireConnection;493;0;491;0
WireConnection;320;1;305;0
WireConnection;320;0;319;0
WireConnection;116;0;107;0
WireConnection;368;0;366;0
WireConnection;353;0;320;0
WireConnection;496;0;116;0
WireConnection;497;0;493;0
WireConnection;372;1;367;0
WireConnection;372;0;368;0
WireConnection;728;1;180;0
WireConnection;450;0;441;0
WireConnection;450;1;442;0
WireConnection;530;0;450;0
WireConnection;62;0;372;0
WireConnection;62;4;742;0
WireConnection;62;7;507;0
WireConnection;62;10;506;0
WireConnection;62;11;354;0
ASEEND*/
//CHKSM=36106B57DF00D246382E0910C92D247408273F49