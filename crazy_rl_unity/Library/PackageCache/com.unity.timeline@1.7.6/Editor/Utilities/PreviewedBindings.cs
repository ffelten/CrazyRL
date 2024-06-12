using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Playables;
using Object = UnityEngine.Object;

namespace UnityEditor.Timeline
{
    readonly struct PreviewedBindings<T> where T : Object
    {
        readonly IEnumerable<T> m_UniqueBindings;
        readonly IReadOnlyDictionary<Object, HashSet<T>> m_BindingsPerObject;

        PreviewedBindings(IEnumerable<T> uniqueBindings, IReadOnlyDictionary<Object, HashSet<T>> bindingsPerObject)
        {
            m_UniqueBindings = uniqueBindings;
            m_BindingsPerObject = bindingsPerObject;
        }

        public IEnumerable<T> GetUniqueBindings() => m_UniqueBindings ?? Enumerable.Empty<T>();

        public IEnumerable<T> GetBindingsForObject(Object track)
        {
            if (m_BindingsPerObject != null && m_BindingsPerObject.TryGetValue(track, out HashSet<T> bindings))
                return bindings;

            return Enumerable.Empty<T>();
        }

        public static PreviewedBindings<T> GetPreviewedBindings(IEnumerable<PlayableDirector> directors)
        {
            var uniqueBindings = new HashSet<T>();
            var bindingsPerTrack = new Dictionary<Object, HashSet<T>>();
            foreach (PlayableDirector director in directors)
            {
                if (director.playableAsset == null) continue;

                foreach (PlayableBinding output in director.playableAsset.outputs)
                {
                    var binding = director.GetGenericBinding(output.sourceObject) as T;
                    Add(output.sourceObject, binding, uniqueBindings, bindingsPerTrack);
                }
            }

            return new PreviewedBindings<T>(uniqueBindings, bindingsPerTrack);
        }

        static void Add(Object obj, T binding, HashSet<T> bindings, Dictionary<Object, HashSet<T>> bindingsPerObject)
        {
            if (binding == null)
                return;

            bindings.Add(binding);
            if (bindingsPerObject.TryGetValue(obj, out HashSet<T> objectBindings))
                objectBindings.Add(binding);
            else
                bindingsPerObject.Add(obj, new HashSet<T> { binding });
        }
    }
}
