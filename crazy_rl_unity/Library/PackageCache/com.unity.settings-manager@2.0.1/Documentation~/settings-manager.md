# Settings Manager

The Settings Manager is a framework that lets you convert any serializable field into a setting, including a pre-built settings interface.

## Installation

To install this package, follow the instructions in the [Package Manager documentation](https://docs.unity3d.com/Manual/upm-ui-install.html).

This package provides a sample that demonstrates how to implement custom user settings. To install them, follow these instructions:

1. Make sure the Settings Manager package is installed in your Unity project. 

2. In the Package Manager window, locate the Settings Manager package select it from the list.

	The [Details view](https://docs.unity3d.com/Manual/upm-ui-details.html) displays information about the Settings Manager package.

3. From the Details view, click the **Import** button under the **Samples** section.

## Requirements

This version of the Settings Manager package is compatible with the following versions of the Unity Editor:

* 2018.4 and later

## Using the Settings Manager

The [Settings](xref:UnityEditor.SettingsManagement.Settings) class is responsible for setting and retrieving serialized values from a settings repository.

Use settings repositories to save and load settings for a specific scope. This package provides two settings repositories: 

* The [UserSettingsRepository](xref:UnityEditor.SettingsManagement.UserSettingsRepository), backed by the [EditorPrefs](xref:UnityEditor.EditorPrefs) class, lets you save [user preferences](https://docs.unity3d.com/Manual/Preferences.html). 
* The [FileSettingsRepository](xref:UnityEditor.SettingsManagement.FileSettingsRepository) saves a JSON file to the `ProjectSettings` directory in order to save [project settings](https://docs.unity3d.com/Manual/comp-ManagerGroup.html).

You can create and manage all settings from a singleton `Settings` instance. For example:

```c#
using UnityEditor.SettingsManagement;

namespace UnityEditor.SettingsManagement.Examples
{
    static class MySettingsManager
    {
        internal const string k_PackageName = "com.example.my-settings-example";

        static Settings s_Instance;

        internal static Settings instance
        {
            get
            {
                if (s_Instance == null)
                    s_Instance = new Settings(k_PackageName);

                return s_Instance;
            }
        }
    }
}
```

### Getting and setting values

Your `Settings` instance should implement generic methods to set and retrieve values:

```
MySettingsManager.instance.Get<float>("myFloatValue", SettingsScope.Project);
```

There are two arguments: key, and scope. The [Settings](xref:UnityEditor.SettingsManagement.Settings) class finds an appropriate [ISettingsRepository](xref:UnityEditor.SettingsManagement.ISettingsRepository) for the scope, while `key` and `T` are used to find the value. Keys are unique among types: you can re-use keys as long as its type is different.

Alternatively, you can use the [UserSetting&lt;T&gt;](xref:UnityEditor.SettingsManagement.UserSetting`1) class to manage settings. This is a wrapper class around the `Settings` get/set properties, which makes it easy to make any field a saved setting.

```c#
// UserSetting<T>(Settings instance, string key, T defaultValue, SettingsScope scope = SettingsScope.Project)
Setting<int> myIntValue = new Setting<int>(MySettingsManager.instance, "int.key", 42, SettingsScope.User);
```

[UserSetting&lt;T&gt;](xref:UnityEditor.SettingsManagement.UserSetting`1) caches the current value, and keeps a copy of the default value so that it may be reset. You can also use `UserSetting<T>` fields with the `[UserSettingAttribute]` attribute, which lets the `SettingsManagerProvider` automatically add it to a settings inspector.

## Settings Provider

To register your settings so they appear in the [Project Settings](https://docs.unity3d.com/Manual/comp-ManagerGroup.html) window, you can either write your own [SettingsProvider](xref:UnityEditor.SettingsProvider) implementation, or use the [UserSettingsProvider](xref:UnityEditor.SettingsManagement.UserSettingsProvider) and let it automatically create your interface.

Making use of `UserSettingsProvider` comes with many benefits, including a uniform look for your settings UI, support for search, and per-field or mass reset support.

```
using UnityEngine;

namespace UnityEditor.SettingsManagement.Examples
{
	static class MySettingsProvider
	{
		const string k_PreferencesPath = "Preferences/My Settings";

		[SettingsProvider]
		static SettingsProvider CreateSettingsProvider()
		{
			// The last parameter tells the provider where to search for settings.
			var provider = new SettingsManagerProvider(k_PreferencesPath,
				MySettingsManager.instance,
				new [] { typeof(MySettingsProvider).Assembly });

			return provider;
		}
	}
}
```

To register a field with the [UserSettingsProvider](xref:UnityEditor.SettingsManagement.UserSettingsProvider), decorate it with `[UserSettingAttribute(string displayCategory, string key)]`. 

> [!NOTE]
> The `[UserSettingAttribute]` decoration is only valid for static fields.

For more complex settings that require additional UI (or that don't have a built-in Editor), use [UserSettingBlockAttribute](xref:UnityEditor.SettingsManagement.UserSettingBlockAttribute) to access the settings provider GUI. For more information, look at the sample source file `SettingsExamples.cs` under the `Assets/Samples/Settings Manager/<version>/User Settings Example/PackageWithProjectAndUserSettings` folder in your Unity project root. 

> [!TIP]
> If you don't see this path or file, follow the steps under the Installation section to import it.
