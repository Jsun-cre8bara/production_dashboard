import React from 'react';

type KpiCardProps = {
  title: string;
  value?: string | number;
  trendText?: string;
};

const KpiCard: React.FC<KpiCardProps> = ({ title, value = '-', trendText = '변동 없음' }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex flex-col space-y-2">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="text-2xl font-semibold text-gray-900">{value}</div>
      <div className="text-xs text-indigo-600">{trendText}</div>
    </div>
  );
};

export default KpiCard;
