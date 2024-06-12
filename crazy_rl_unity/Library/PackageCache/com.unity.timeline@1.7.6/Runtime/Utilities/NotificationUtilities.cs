using System;
using System.Collections.Generic;
using UnityEngine.Playables;

namespace UnityEngine.Timeline
{
    static class NotificationUtilities
    {
        public static ScriptPlayable<TimeNotificationBehaviour> CreateNotificationsPlayable(PlayableGraph graph, IEnumerable<IMarker> markers, PlayableDirector director)
        {
            return CreateNotificationsPlayable(graph, markers, null, director);
        }

        public static ScriptPlayable<TimeNotificationBehaviour> CreateNotificationsPlayable(PlayableGraph graph, IEnumerable<IMarker> markers, TimelineAsset timelineAsset)
        {
            return CreateNotificationsPlayable(graph, markers, timelineAsset, null);
        }

        static ScriptPlayable<TimeNotificationBehaviour> CreateNotificationsPlayable(PlayableGraph graph, IEnumerable<IMarker> markers, IPlayableAsset asset, PlayableDirector director)
        {
            ScriptPlayable<TimeNotificationBehaviour> notificationPlayable = ScriptPlayable<TimeNotificationBehaviour>.Null;
            DirectorWrapMode extrapolationMode = director != null ? director.extrapolationMode : DirectorWrapMode.None;
            bool didCalculateDuration = false;
            var duration = 0d;

            foreach (IMarker e in markers)
            {
                var notification = e as INotification;
                if (notification == null)
                    continue;

                if (!didCalculateDuration)
                {
                    duration = director != null ? director.playableAsset.duration : asset.duration;
                    didCalculateDuration = true;
                }

                if (notificationPlayable.Equals(ScriptPlayable<TimeNotificationBehaviour>.Null))
                {
                    notificationPlayable = TimeNotificationBehaviour.Create(graph,
                        duration, extrapolationMode);
                }

                var time = (DiscreteTime)e.time;
                var tlDuration = (DiscreteTime)duration;
                if (time >= tlDuration && time <= tlDuration.OneTickAfter() && tlDuration != 0)
                    time = tlDuration.OneTickBefore();

                if (e is INotificationOptionProvider notificationOptionProvider)
                    notificationPlayable.GetBehaviour().AddNotification((double)time, notification, notificationOptionProvider.flags);
                else
                    notificationPlayable.GetBehaviour().AddNotification((double)time, notification);
            }

            return notificationPlayable;
        }

        public static bool TrackTypeSupportsNotifications(Type type)
        {
            var binding = (TrackBindingTypeAttribute)Attribute.GetCustomAttribute(type, typeof(TrackBindingTypeAttribute));
            return binding != null &&
                (typeof(Component).IsAssignableFrom(binding.type) ||
                    typeof(GameObject).IsAssignableFrom(binding.type));
        }
    }
}
