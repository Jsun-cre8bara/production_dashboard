import React from 'react';

const menuItems = [
  'Dashboard',
  '공연 관리',
  '티켓·좌석 관리',
  '관객 데이터 분석',
  '정산·매출 리포트',
  '쿠폰·프로모션',
  '설정',
];

const Sidebar: React.FC = () => {
  return (
    <aside className="w-60 min-h-screen bg-gray-900 text-gray-200 flex flex-col py-8 px-4">
      <nav className="space-y-2">
        {menuItems.map((item) => (
          <button
            key={item}
            className={`w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
              item === 'Dashboard' ? 'bg-gray-800 text-white' : 'hover:bg-gray-800 hover:text-white'
            }`}
          >
            {item}
          </button>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
