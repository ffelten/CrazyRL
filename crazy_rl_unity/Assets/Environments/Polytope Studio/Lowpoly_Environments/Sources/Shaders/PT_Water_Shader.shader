// Made with Amplify Shader Editor v1.9.1.5
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Polytope Studio/PT_Water_Shader"
{
	Properties
	{
		_DeepColor("Deep Color", Color) = (0.3114988,0.5266015,0.5283019,0)
		_ShallowColor("Shallow Color", Color) = (0.5238074,0.7314408,0.745283,0)
		_Depth("Depth", Range( 0 , 1)) = 0.3
		_DepthStrength("Depth Strength", Range( 0 , 1)) = 0.3
		_Smootness("Smootness", Range( 0 , 1)) = 1
		_Mettalic("Mettalic", Range( 0 , 1)) = 1
		_TessValue( "Max Tessellation", Range( 1, 32 ) ) = 5
		_WaveSpeed("Wave Speed", Range( 0 , 1)) = 0.5
		_WaveTile("Wave Tile", Range( 0 , 0.9)) = 0.5
		_WaveAmplitude("Wave Amplitude", Range( 0 , 1)) = 0.2
		[NoScaleOffset][Normal]_NormalMapTexture("Normal Map Texture ", 2D) = "bump" {}
		_NormalMapWavesSpeed("Normal Map Waves Speed", Range( 0 , 1)) = 0.1
		_NormalMapsWavesSize("Normal Maps Waves Size", Range( 0 , 10)) = 5
		_FoamColor("Foam Color", Color) = (0.3066038,1,0.9145772,0)
		_FoamAmount("Foam Amount", Range( 0 , 10)) = 1.5
		_FoamPower("Foam Power", Range( 0.1 , 5)) = 0.5
		_FoamNoiseScale("Foam Noise Scale", Range( 0 , 1000)) = 150
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
		[Header(Forward Rendering Options)]
		[ToggleOff] _GlossyReflections("Reflections", Float) = 1.0
	}

	SubShader
	{
		Tags{ "RenderType" = "Transparent"  "Queue" = "Transparent+0" "IgnoreProjector" = "True" }
		Cull Off
		GrabPass{ }
		CGPROGRAM
		#include "UnityShaderVariables.cginc"
		#include "UnityCG.cginc"
		#pragma target 5.0
		#pragma shader_feature _GLOSSYREFLECTIONS_OFF
		#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)
		#define ASE_DECLARE_SCREENSPACE_TEXTURE(tex) UNITY_DECLARE_SCREENSPACE_TEXTURE(tex);
		#else
		#define ASE_DECLARE_SCREENSPACE_TEXTURE(tex) UNITY_DECLARE_SCREENSPACE_TEXTURE(tex)
		#endif
		#pragma surface surf Standard alpha:fade keepalpha noshadow vertex:vertexDataFunc tessellate:tessFunction 
		struct Input
		{
			float3 worldPos;
			float4 screenPos;
			float2 uv_texcoord;
		};

		uniform float _WaveAmplitude;
		uniform float _WaveSpeed;
		uniform float _WaveTile;
		UNITY_DECLARE_DEPTH_TEXTURE( _CameraDepthTexture );
		uniform float4 _CameraDepthTexture_TexelSize;
		uniform float _FoamAmount;
		uniform float _FoamPower;
		uniform float _FoamNoiseScale;
		uniform sampler2D _NormalMapTexture;
		uniform float _NormalMapsWavesSize;
		uniform float _NormalMapWavesSpeed;
		ASE_DECLARE_SCREENSPACE_TEXTURE( _GrabTexture )
		uniform float4 _ShallowColor;
		uniform float4 _DeepColor;
		uniform float _DepthStrength;
		uniform float _Depth;
		uniform float4 _FoamColor;
		uniform float _Mettalic;
		uniform float _Smootness;
		uniform float _TessValue;


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


		float2 UnityGradientNoiseDir( float2 p )
		{
			p = fmod(p , 289);
			float x = fmod((34 * p.x + 1) * p.x , 289) + p.y;
			x = fmod( (34 * x + 1) * x , 289);
			x = frac( x / 41 ) * 2 - 1;
			return normalize( float2(x - floor(x + 0.5 ), abs( x ) - 0.5 ) );
		}
		
		float UnityGradientNoise( float2 UV, float Scale )
		{
			float2 p = UV * Scale;
			float2 ip = floor( p );
			float2 fp = frac( p );
			float d00 = dot( UnityGradientNoiseDir( ip ), fp );
			float d01 = dot( UnityGradientNoiseDir( ip + float2( 0, 1 ) ), fp - float2( 0, 1 ) );
			float d10 = dot( UnityGradientNoiseDir( ip + float2( 1, 0 ) ), fp - float2( 1, 0 ) );
			float d11 = dot( UnityGradientNoiseDir( ip + float2( 1, 1 ) ), fp - float2( 1, 1 ) );
			fp = fp * fp * fp * ( fp * ( fp * 6 - 15 ) + 10 );
			return lerp( lerp( d00, d01, fp.y ), lerp( d10, d11, fp.y ), fp.x ) + 0.5;
		}


		inline float4 ASE_ComputeGrabScreenPos( float4 pos )
		{
			#if UNITY_UV_STARTS_AT_TOP
			float scale = -1.0;
			#else
			float scale = 1.0;
			#endif
			float4 o = pos;
			o.y = pos.w * 0.5f;
			o.y = ( pos.y - o.y ) * _ProjectionParams.x * scale + o.y;
			return o;
		}


		float4 tessFunction( )
		{
			return _TessValue;
		}

		void vertexDataFunc( inout appdata_full v )
		{
			float4 appendResult153 = (float4(0.23 , -0.8 , 0.0 , 0.0));
			float3 ase_worldPos = mul( unity_ObjectToWorld, v.vertex );
			float4 appendResult156 = (float4(ase_worldPos.x , ase_worldPos.z , 0.0 , 0.0));
			float2 panner145 = ( ( _Time.y * _WaveSpeed ) * appendResult153.xy + ( ( appendResult156 * float4( float2( 6.5,0.9 ), 0.0 , 0.0 ) ) * _WaveTile ).xy);
			float simplePerlin2D143 = snoise( panner145 );
			simplePerlin2D143 = simplePerlin2D143*0.5 + 0.5;
			float WAVESDISPLACEMENT245 = ( ( float3(0,0.05,0).y * _WaveAmplitude ) * simplePerlin2D143 );
			float3 temp_cast_3 = (WAVESDISPLACEMENT245).xxx;
			v.vertex.xyz += temp_cast_3;
			v.vertex.w = 1;
		}

		void surf( Input i , inout SurfaceOutputStandard o )
		{
			float4 ase_screenPos = float4( i.screenPos.xyz , i.screenPos.w + 0.00000000001 );
			float4 ase_screenPosNorm = ase_screenPos / ase_screenPos.w;
			ase_screenPosNorm.z = ( UNITY_NEAR_CLIP_VALUE >= 0 ) ? ase_screenPosNorm.z : ase_screenPosNorm.z * 0.5 + 0.5;
			float screenDepth434 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE( _CameraDepthTexture, ase_screenPosNorm.xy ));
			float distanceDepth434 = abs( ( screenDepth434 - LinearEyeDepth( ase_screenPosNorm.z ) ) / ( _FoamAmount ) );
			float saferPower436 = abs( distanceDepth434 );
			float temp_output_436_0 = pow( saferPower436 , _FoamPower );
			float2 temp_cast_0 = (_FoamNoiseScale).xx;
			float2 temp_cast_1 = (( _Time.y * 0.2 )).xx;
			float2 uv_TexCoord433 = i.uv_texcoord * temp_cast_0 + temp_cast_1;
			float gradientNoise437 = UnityGradientNoise(uv_TexCoord433,1.0);
			gradientNoise437 = gradientNoise437*0.5 + 0.5;
			float temp_output_471_0 = step( temp_output_436_0 , gradientNoise437 );
			float FoamMask439 = temp_output_471_0;
			float4 appendResult405 = (float4(_NormalMapsWavesSize , _NormalMapsWavesSize , 0.0 , 0.0));
			float mulTime251 = _Time.y * 0.1;
			float2 temp_cast_3 = (( mulTime251 * _NormalMapWavesSpeed )).xx;
			float2 uv_TexCoord254 = i.uv_texcoord * appendResult405.xy + temp_cast_3;
			float2 temp_output_2_0_g9 = uv_TexCoord254;
			float2 break6_g9 = temp_output_2_0_g9;
			float temp_output_25_0_g9 = ( pow( 0.5 , 3.0 ) * 0.1 );
			float2 appendResult8_g9 = (float2(( break6_g9.x + temp_output_25_0_g9 ) , break6_g9.y));
			float4 tex2DNode14_g9 = tex2D( _NormalMapTexture, temp_output_2_0_g9 );
			float temp_output_4_0_g9 = 1.0;
			float3 appendResult13_g9 = (float3(1.0 , 0.0 , ( ( tex2D( _NormalMapTexture, appendResult8_g9 ).g - tex2DNode14_g9.g ) * temp_output_4_0_g9 )));
			float2 appendResult9_g9 = (float2(break6_g9.x , ( break6_g9.y + temp_output_25_0_g9 )));
			float3 appendResult16_g9 = (float3(0.0 , 1.0 , ( ( tex2D( _NormalMapTexture, appendResult9_g9 ).g - tex2DNode14_g9.g ) * temp_output_4_0_g9 )));
			float3 normalizeResult22_g9 = normalize( cross( appendResult13_g9 , appendResult16_g9 ) );
			float3 NORMALMAPWAVES243 = normalizeResult22_g9;
			float4 color478 = IsGammaSpace() ? float4(0.4980392,0.4980392,1,0) : float4(0.2122307,0.2122307,1,0);
			float layeredBlendVar477 = FoamMask439;
			float4 layeredBlend477 = ( lerp( float4( NORMALMAPWAVES243 , 0.0 ),color478 , layeredBlendVar477 ) );
			float4 normalizeResult474 = normalize( layeredBlend477 );
			o.Normal = normalizeResult474.rgb;
			float screenDepth350 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE( _CameraDepthTexture, ase_screenPosNorm.xy ));
			float distanceDepth350 = abs( ( screenDepth350 - LinearEyeDepth( ase_screenPosNorm.z ) ) / ( 100.0 ) );
			float4 ase_grabScreenPos = ASE_ComputeGrabScreenPos( ase_screenPos );
			float4 ase_grabScreenPosNorm = ase_grabScreenPos / ase_grabScreenPos.w;
			float4 screenColor314 = UNITY_SAMPLE_SCREENSPACE_TEXTURE(_GrabTexture,( (ase_grabScreenPosNorm).xyzw + float4( ( NORMALMAPWAVES243 * 1.0 ) , 0.0 ) ).xy);
			float4 FAKEREFRACTIONS415 = ( ( 1.0 - distanceDepth350 ) * screenColor314 );
			float eyeDepth64 = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE( _CameraDepthTexture, ase_screenPosNorm.xy ));
			float clampResult81 = clamp( ( _DepthStrength * ( eyeDepth64 - ( ase_screenPos.w + _Depth ) ) ) , 0.0 , 1.0 );
			float4 lerpResult86 = lerp( _ShallowColor , _DeepColor , clampResult81);
			float4 DeepShallowColor196 = lerpResult86;
			float4 lerpResult470 = lerp( FAKEREFRACTIONS415 , DeepShallowColor196 , float4( 0.6132076,0.6132076,0.6132076,0 ));
			float4 FoamColor442 = ( temp_output_471_0 * _FoamColor );
			o.Albedo = ( lerpResult470 + FoamColor442 ).rgb;
			o.Metallic = _Mettalic;
			float4 temp_cast_9 = (_Smootness).xxxx;
			float4 color484 = IsGammaSpace() ? float4(0.2264151,0.2264151,0.2264151,0) : float4(0.04193995,0.04193995,0.04193995,0);
			float layeredBlendVar485 = FoamMask439;
			float4 layeredBlend485 = ( lerp( temp_cast_9,color484 , layeredBlendVar485 ) );
			o.Smoothness = layeredBlend485.r;
			float DeepShallowMask197 = clampResult81;
			float smoothstepResult400 = smoothstep( 0.2 , 1.2 , FoamMask439);
			float clampResult401 = clamp( ( smoothstepResult400 * 0.05 ) , 0.0 , 1.0 );
			float TRANSPARENCYFINAL267 = ( DeepShallowMask197 + (1.0 + (0.95 - 0.0) * (0.0 - 1.0) / (1.0 - 0.0)) + clampResult401 );
			o.Alpha = TRANSPARENCYFINAL267;
		}

		ENDCG
	}
}
/*ASEBEGIN
Version=19105
Node;AmplifyShaderEditor.CommentaryNode;248;-2607.635,-1748.536;Inherit;False;1847.655;579.5157;Comment;9;243;352;254;405;249;257;251;252;351;Normal Map Waves;1,1,1,1;0;0
Node;AmplifyShaderEditor.CommentaryNode;428;-65.96826,-1461.15;Inherit;False;1745.648;737.933;Foam;16;445;443;442;441;440;439;437;436;435;434;433;432;431;430;429;471;Foam;1,1,1,1;0;0
Node;AmplifyShaderEditor.SimpleTimeNode;251;-2324.447,-1362.269;Inherit;False;1;0;FLOAT;0.1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;257;-2561.399,-1535.609;Inherit;False;Property;_NormalMapsWavesSize;Normal Maps Waves Size;18;0;Create;True;0;0;0;False;0;False;5;5;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;252;-2485.628,-1264.38;Inherit;False;Property;_NormalMapWavesSpeed;Normal Map Waves Speed;17;0;Create;True;0;0;0;False;0;False;0.1;0.1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.DynamicAppendNode;405;-2190.042,-1513.174;Inherit;False;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;249;-2082.218,-1380.983;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;445;-71.74513,-811.948;Inherit;False;Constant;_FoamSpeed;Foam Speed;20;0;Create;True;0;0;0;False;0;False;0.2;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleTimeNode;429;-17.23541,-917.0339;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;430;-26.14526,-1064.283;Inherit;False;Property;_FoamNoiseScale;Foam Noise Scale;22;0;Create;True;0;0;0;False;0;False;150;150;0;1000;0;1;FLOAT;0
Node;AmplifyShaderEditor.TexturePropertyNode;351;-2189.521,-1740.015;Inherit;True;Property;_NormalMapTexture;Normal Map Texture ;16;2;[NoScaleOffset];[Normal];Create;True;0;0;0;True;0;False;a78adb8868cccbe4a92d9d81db916e6e;a78adb8868cccbe4a92d9d81db916e6e;True;bump;Auto;Texture2D;-1;0;2;SAMPLER2D;0;SAMPLERSTATE;1
Node;AmplifyShaderEditor.RangedFloatNode;431;-27.87524,-1268.683;Inherit;False;Property;_FoamAmount;Foam Amount;20;0;Create;True;0;0;0;False;0;False;1.5;1.5;0;10;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;91;-2474.408,466.2085;Inherit;False;1982.289;589.8825;Comment;14;63;65;67;64;83;82;66;74;81;85;86;84;197;196;Deep&ShallowColor;1,1,1,1;0;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;254;-1992.74,-1514.106;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,5;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;432;227.9714,-894.1125;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;435;256.7495,-1206.154;Inherit;False;Property;_FoamPower;Foam Power;21;0;Create;True;0;0;0;False;0;False;0.5;0.5;0.1;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.ScreenPosInputsNode;63;-2442.563,708.9407;Float;False;1;False;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.FunctionNode;352;-1730.17,-1725.82;Inherit;True;NormalCreate;0;;9;e12f7ae19d416b942820e3932b56220f;0;4;1;SAMPLER2D;;False;2;FLOAT2;0,0;False;3;FLOAT;0.5;False;4;FLOAT;1;False;1;FLOAT3;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;433;290.627,-1062.13;Inherit;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.DepthFade;434;224.4038,-1418.474;Inherit;False;True;False;True;2;1;FLOAT3;0,0,0;False;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;183;-2427.607,1285.62;Inherit;False;1957.563;1136.648;Comment;21;155;156;158;151;148;150;152;157;160;149;159;153;145;162;168;170;143;169;163;188;245;WavesVertexOffset;1,1,1,1;0;0
Node;AmplifyShaderEditor.RangedFloatNode;67;-2355.368,899.1294;Inherit;False;Property;_Depth;Depth;4;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.PowerNode;436;594.9379,-1353.827;Inherit;False;True;2;0;FLOAT;0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;437;667.4668,-1066.86;Inherit;True;Gradient;True;True;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.BreakToComponentsNode;65;-2215.166,717.8157;Inherit;False;FLOAT4;1;0;FLOAT4;0,0,0,0;False;16;FLOAT;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4;FLOAT;5;FLOAT;6;FLOAT;7;FLOAT;8;FLOAT;9;FLOAT;10;FLOAT;11;FLOAT;12;FLOAT;13;FLOAT;14;FLOAT;15
Node;AmplifyShaderEditor.CommentaryNode;414;-2408.578,-776.7009;Inherit;False;1714.404;696.3644;Comment;12;290;342;280;333;279;341;415;335;336;314;350;282;FAKE REFRACTIONS;1,1,1,1;0;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;243;-1071.836,-1714.044;Inherit;False;NORMALMAPWAVES;-1;True;1;0;FLOAT3;0,0,0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.GetLocalVarNode;279;-2170.569,-303.5186;Inherit;False;243;NORMALMAPWAVES;1;0;OBJECT;;False;1;FLOAT3;0
Node;AmplifyShaderEditor.StepOpNode;471;841.0071,-1233.626;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;83;-2088.616,791.182;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GrabScreenPosition;341;-2358.578,-519.3195;Inherit;False;0;0;5;FLOAT4;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;290;-2208.454,-202.0541;Inherit;False;Constant;_Float3;Float 3;18;0;Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.WorldPosInputsNode;155;-2377.607,1537.926;Inherit;False;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.ScreenDepthNode;64;-2258.576,616.881;Inherit;False;0;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;333;-1985.109,-686.3021;Inherit;False;Constant;_Float4;Float 4;22;0;Create;True;0;0;0;False;0;False;100;1;0;100;0;1;FLOAT;0
Node;AmplifyShaderEditor.CommentaryNode;266;170.6061,898.9609;Inherit;False;1198.132;582.9205;Comment;9;3;192;90;265;198;267;398;400;401;TRANSPARENCY FINAL;1,1,1,1;0;0
Node;AmplifyShaderEditor.SimpleSubtractOpNode;66;-1956.295,733.16;Inherit;False;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;439;1437.562,-1352.665;Inherit;False;FoamMask;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;82;-2058.965,595.7329;Inherit;False;Property;_DepthStrength;Depth Strength;5;0;Create;True;0;0;0;False;0;False;0.3;0.3;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.Vector2Node;158;-2163.74,1704.801;Inherit;False;Constant;_Vector0;Vector 0;15;0;Create;True;0;0;0;False;0;False;6.5,0.9;1,10;0;3;FLOAT2;0;FLOAT;1;FLOAT;2
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;280;-1875.663,-306.6546;Inherit;False;2;2;0;FLOAT3;0,0,0;False;1;FLOAT;0;False;1;FLOAT3;0
Node;AmplifyShaderEditor.DynamicAppendNode;156;-2155.255,1544.367;Inherit;False;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.ComponentMaskNode;342;-2090.173,-513.0981;Inherit;False;True;True;True;True;1;0;FLOAT4;0,0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;160;-1960.265,1772.124;Inherit;False;Property;_WaveTile;Wave Tile;14;0;Create;True;0;0;0;False;0;False;0.5;0.5;0;0.9;0;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;151;-2243.51,1996.039;Inherit;False;Constant;_wavedirectionx;wave direction x;17;0;Create;True;0;0;0;False;0;False;0.23;-1;-1;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;282;-1591.159,-357.7633;Inherit;False;2;2;0;FLOAT4;0,0,0,0;False;1;FLOAT3;0,0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.DepthFade;350;-1684.137,-726.7009;Inherit;False;True;False;True;2;1;FLOAT3;0,0,0;False;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;150;-1855.533,2306.268;Inherit;False;Property;_WaveSpeed;Wave Speed;13;0;Create;True;0;0;0;False;0;False;0.5;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;74;-1761.804,708.0941;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;157;-1896.241,1546.953;Inherit;False;2;2;0;FLOAT4;0,0,0,0;False;1;FLOAT2;0,0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.SimpleTimeNode;148;-1754.768,2203.957;Inherit;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;192;41.23421,1217.474;Inherit;False;439;FoamMask;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;152;-2236.342,2081.59;Inherit;False;Constant;_wavedirectiony;wave direction y;24;0;Create;True;0;0;0;False;0;False;-0.8;0.067;-1;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ClampOpNode;81;-1596.362,711.3876;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;149;-1530.768,2253.957;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;159;-1643.2,1549.363;Inherit;False;2;2;0;FLOAT4;0,0,0,0;False;1;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.ScreenColorNode;314;-1416.112,-356.7128;Inherit;False;Global;_GrabScreen0;Grab Screen 0;21;0;Create;True;0;0;0;False;0;False;Object;-1;False;False;False;False;2;0;FLOAT2;0,0;False;1;FLOAT;0;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.Vector3Node;162;-1513.271,1361.713;Inherit;False;Constant;_Vector1;Vector 1;14;0;Create;True;0;0;0;False;0;False;0,0.05,0;0,0,0;0;4;FLOAT3;0;FLOAT;1;FLOAT;2;FLOAT;3
Node;AmplifyShaderEditor.DynamicAppendNode;153;-1867.231,2019.957;Inherit;False;FLOAT4;4;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;3;FLOAT;0;False;1;FLOAT4;0
Node;AmplifyShaderEditor.RangedFloatNode;399;277.303,1436.751;Inherit;False;Constant;_Float0;Float 0;29;0;Create;True;0;0;0;True;0;False;0.05;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;85;-1353.423,608.9352;Inherit;False;Property;_ShallowColor;Shallow Color;3;0;Create;True;0;0;0;False;0;False;0.5238074,0.7314408,0.745283,0;0.2003827,0.745283,0.3747112,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.OneMinusNode;335;-1387.788,-726.2308;Inherit;False;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;84;-1327.132,818.0287;Inherit;False;Property;_DeepColor;Deep Color;2;0;Create;True;0;0;0;False;0;False;0.3114988,0.5266015,0.5283019,0;1,0.8726415,0.8726415,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SmoothstepOpNode;400;362.052,1219.602;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0.2;False;2;FLOAT;1.2;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;398;577.6862,1288.293;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;336;-1165.583,-388.4095;Inherit;True;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.LerpOp;86;-1045.415,672.1371;Inherit;False;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.PannerNode;145;-1386.378,1988.409;Inherit;False;3;0;FLOAT2;0,0;False;2;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT2;0
Node;AmplifyShaderEditor.RangedFloatNode;3;193.7791,1014.227;Inherit;False;Constant;_Transparency;Transparency;2;0;Create;True;0;0;0;False;0;False;0.95;0.5;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.BreakToComponentsNode;168;-1298.1,1387.307;Inherit;False;FLOAT3;1;0;FLOAT3;0,0,0;False;16;FLOAT;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4;FLOAT;5;FLOAT;6;FLOAT;7;FLOAT;8;FLOAT;9;FLOAT;10;FLOAT;11;FLOAT;12;FLOAT;13;FLOAT;14;FLOAT;15
Node;AmplifyShaderEditor.RangedFloatNode;170;-1409.1,1596.307;Inherit;False;Property;_WaveAmplitude;Wave Amplitude;15;0;Create;True;0;0;0;False;0;False;0.2;0.2;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;440;999.5128,-922.9009;Inherit;False;Property;_FoamColor;Foam Color;19;0;Create;True;0;0;0;False;0;False;0.3066038,1,0.9145772,0;0.2971698,1,0.9126425,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RegisterLocalVarNode;197;-1377.094,528.7972;Inherit;False;DeepShallowMask;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ClampOpNode;401;715.3116,1244.153;Inherit;False;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.NoiseGeneratorNode;143;-1126.841,1911.796;Inherit;True;Simplex2D;True;False;2;0;FLOAT2;0,0;False;1;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;415;-925.225,-389.6505;Inherit;False;FAKEREFRACTIONS;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;198;494.371,948.9609;Inherit;False;197;DeepShallowMask;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;441;1249.635,-1154.369;Inherit;False;2;2;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.TFHCRemapNode;265;508.0719,1046.37;Inherit;False;5;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;1;False;3;FLOAT;1;False;4;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;196;-731.6638,665.9779;Inherit;False;DeepShallowColor;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;169;-1167.1,1475.307;Inherit;False;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;416;909.8789,-371.3734;Inherit;False;415;FAKEREFRACTIONS;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;476;1045.112,-63.59784;Inherit;False;439;FoamMask;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;90;833.869,1012.861;Inherit;False;3;3;0;FLOAT;0;False;1;FLOAT;0;False;2;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.ColorNode;478;1017.163,100.7891;Inherit;False;Constant;_Color0;Color 0;15;0;Create;True;0;0;0;False;0;False;0.4980392,0.4980392,1,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.GetLocalVarNode;411;903.5338,-288.2307;Inherit;False;196;DeepShallowColor;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.GetLocalVarNode;457;1005.23,19.06134;Inherit;False;243;NORMALMAPWAVES;1;0;OBJECT;;False;1;FLOAT3;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;442;1435.016,-1148.227;Inherit;False;FoamColor;-1;True;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;163;-963.286,1360.345;Inherit;True;2;2;0;FLOAT;0;False;1;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;483;1068.577,311.6291;Inherit;False;439;FoamMask;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.RangedFloatNode;1;1085.717,394.7116;Inherit;False;Property;_Smootness;Smootness;6;0;Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;460;1400.639,-169.8729;Inherit;False;442;FoamColor;1;0;OBJECT;;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;267;1047.772,1004.04;Inherit;False;TRANSPARENCYFINAL;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LayeredBlendNode;477;1422.112,-51.59784;Inherit;False;6;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.ColorNode;484;1118.628,475.016;Inherit;False;Constant;_Color1;Color 1;15;0;Create;True;0;0;0;False;0;False;0.2264151,0.2264151,0.2264151,0;0,0,0,0;True;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.LerpOp;470;1194.726,-397.0083;Inherit;True;3;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;2;COLOR;0.6132076,0.6132076,0.6132076,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;245;-729.6497,1371.044;Inherit;False;WAVESDISPLACEMENT;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleAddOpNode;468;1626.736,-390.0578;Inherit;True;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;16;1430.962,91.4223;Inherit;False;Property;_Mettalic;Mettalic;7;0;Create;True;0;0;0;False;0;False;1;1;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;246;1792.341,447.4535;Inherit;False;245;WAVESDISPLACEMENT;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;443;802.7948,-1429.559;Inherit;False;newfoammask;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.GetLocalVarNode;268;1802.455,333.9038;Inherit;False;267;TRANSPARENCYFINAL;1;0;OBJECT;;False;1;FLOAT;0
Node;AmplifyShaderEditor.RegisterLocalVarNode;188;-830.4629,1926.917;Inherit;False;OffsetWavesMask;-1;True;1;0;FLOAT;0;False;1;FLOAT;0
Node;AmplifyShaderEditor.LayeredBlendNode;485;1425.577,305.6291;Inherit;False;6;0;FLOAT;0;False;1;COLOR;0,0,0,0;False;2;COLOR;0,0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.NormalizeNode;474;1650.911,-43.49784;Inherit;False;False;1;0;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;0;2122.445,-57.32889;Float;False;True;-1;7;;0;0;Standard;Polytope Studio/PT_Water_Shader;False;False;False;False;False;False;False;False;False;False;False;False;False;False;True;False;False;False;False;True;False;Off;1;False;;0;False;;False;0;False;;0;False;;False;0;Transparent;0.5;True;False;0;False;Transparent;;Transparent;All;12;all;True;True;True;True;0;False;;False;0;False;;255;False;;255;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;0;False;;True;1;5;1;10;False;0.52;False;2;5;False;;10;False;;0;0;False;;0;False;;0;False;;0;False;;1;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;True;Relative;0;;-1;-1;-1;8;0;False;0;0;False;;-1;0;False;;0;0;0;False;0.1;False;;0;False;;False;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
WireConnection;405;0;257;0
WireConnection;405;1;257;0
WireConnection;249;0;251;0
WireConnection;249;1;252;0
WireConnection;254;0;405;0
WireConnection;254;1;249;0
WireConnection;432;0;429;0
WireConnection;432;1;445;0
WireConnection;352;1;351;0
WireConnection;352;2;254;0
WireConnection;433;0;430;0
WireConnection;433;1;432;0
WireConnection;434;0;431;0
WireConnection;436;0;434;0
WireConnection;436;1;435;0
WireConnection;437;0;433;0
WireConnection;65;0;63;0
WireConnection;243;0;352;0
WireConnection;471;0;436;0
WireConnection;471;1;437;0
WireConnection;83;0;65;3
WireConnection;83;1;67;0
WireConnection;66;0;64;0
WireConnection;66;1;83;0
WireConnection;439;0;471;0
WireConnection;280;0;279;0
WireConnection;280;1;290;0
WireConnection;156;0;155;1
WireConnection;156;1;155;3
WireConnection;342;0;341;0
WireConnection;282;0;342;0
WireConnection;282;1;280;0
WireConnection;350;0;333;0
WireConnection;74;0;82;0
WireConnection;74;1;66;0
WireConnection;157;0;156;0
WireConnection;157;1;158;0
WireConnection;81;0;74;0
WireConnection;149;0;148;0
WireConnection;149;1;150;0
WireConnection;159;0;157;0
WireConnection;159;1;160;0
WireConnection;314;0;282;0
WireConnection;153;0;151;0
WireConnection;153;1;152;0
WireConnection;335;0;350;0
WireConnection;400;0;192;0
WireConnection;398;0;400;0
WireConnection;398;1;399;0
WireConnection;336;0;335;0
WireConnection;336;1;314;0
WireConnection;86;0;85;0
WireConnection;86;1;84;0
WireConnection;86;2;81;0
WireConnection;145;0;159;0
WireConnection;145;2;153;0
WireConnection;145;1;149;0
WireConnection;168;0;162;0
WireConnection;197;0;81;0
WireConnection;401;0;398;0
WireConnection;143;0;145;0
WireConnection;415;0;336;0
WireConnection;441;0;471;0
WireConnection;441;1;440;0
WireConnection;265;0;3;0
WireConnection;196;0;86;0
WireConnection;169;0;168;1
WireConnection;169;1;170;0
WireConnection;90;0;198;0
WireConnection;90;1;265;0
WireConnection;90;2;401;0
WireConnection;442;0;441;0
WireConnection;163;0;169;0
WireConnection;163;1;143;0
WireConnection;267;0;90;0
WireConnection;477;0;476;0
WireConnection;477;1;457;0
WireConnection;477;2;478;0
WireConnection;470;0;416;0
WireConnection;470;1;411;0
WireConnection;245;0;163;0
WireConnection;468;0;470;0
WireConnection;468;1;460;0
WireConnection;443;0;436;0
WireConnection;188;0;143;0
WireConnection;485;0;483;0
WireConnection;485;1;1;0
WireConnection;485;2;484;0
WireConnection;474;0;477;0
WireConnection;0;0;468;0
WireConnection;0;1;474;0
WireConnection;0;3;16;0
WireConnection;0;4;485;0
WireConnection;0;9;268;0
WireConnection;0;11;246;0
ASEEND*/
//CHKSM=355FE5F91F7945669292C5BA76537DBE19F7B140