import React from 'react';
import { motion } from 'framer-motion';

export default function StatsRibbon({ data }) {
  const total = data.length;
  const severe = data.filter(d => d.severity === 'SEVERE').length;
  const moderate = data.filter(d => d.severity === 'MODERATE').length;
  const maxDepth = Math.max(...data.map(d => d.depth_cm));

  const stats = [
    { value: total, label: 'Hazards Tracked', colorClass: 'orange' },
    { value: severe, label: 'Critical Craters', colorClass: 'critical' },
    { value: moderate, label: 'Moderate Defects', colorClass: 'moderate' },
    { value: `${maxDepth}cm`, label: 'Max Depression', colorClass: 'critical' },
  ];

  return (
    <div className="stats-ribbon">
      {stats.map((stat, i) => (
        <motion.div
          key={i}
          className="stat-cell"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: i * 0.1 }}
        >
          <div className={`stat-value ${stat.colorClass}`}>{stat.value}</div>
          <div className="stat-label">{stat.label}</div>
        </motion.div>
      ))}
    </div>
  );
}
