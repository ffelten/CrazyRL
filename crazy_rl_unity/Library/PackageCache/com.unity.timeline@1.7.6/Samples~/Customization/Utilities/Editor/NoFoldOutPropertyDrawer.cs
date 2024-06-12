using UnityEditor;
using UnityEngine;

namespace Timeline.Samples
{
    // Custom property drawer that draws all child properties inline
    [CustomPropertyDrawer(typeof(NoFoldOutAttribute))]
    public class NoFoldOutPropertyDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (!property.hasChildren)
                return base.GetPropertyHeight(property, label);
            property.isExpanded = true;
            return EditorGUI.GetPropertyHeight(property, label, true) -
                EditorGUI.GetPropertyHeight(property, label, false);
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (!property.hasChildren)
                EditorGUI.PropertyField(position, property, label);
            else
            {
                SerializedProperty iter = property.Copy();
                var nextSibling = property.Copy();
                nextSibling.Next(false);
                property.Next(true);
                do
                {
                    // We need to check against nextSibling to properly stop
                    // otherwise we will draw properties that are not child of this
                    // foldout.
                    if (SerializedProperty.EqualContents(property, nextSibling))
                        break;
                    float height = EditorGUI.GetPropertyHeight(property, property.hasVisibleChildren);
                    position.height = height;
                    EditorGUI.PropertyField(position, property, property.hasVisibleChildren);
                    position.y = position.y + height;
                }
                while (property.NextVisible(false));
            }
        }
    }
}
